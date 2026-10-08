"""How decorrelated trained models are, and which of them make the best ensemble.

Every model is read once per partition on the Kaggle cut (`--grouping kaggle`, scored exactly as `ioi` scores it: the
unseen-span words of every counted domain, tau chosen on the ensemble's own validation macro). Its context-head
log-probabilities are kept; an ensemble is then a tensor average in log-probability space, the geometric mean
`Ensemble` computes inside the network.

Decorrelation is the Pearson correlation, over the pooled scored words, of each model's log-probability of the TRUE
class. It is continuous (a top-10 hit indicator is binary and saturates near the ceiling of the ruler), and it is the
quantity whose disagreement an average can exploit: two models with equal bAcc@10 but uncorrelated true-class
log-probabilities average to a better one. Pairs are grouped by recipe: same recipe different seed (`SEED`), same
scheme (leading name token) different knob (`KNOB`), different scheme (`SCHEME`).

Greedy forward selection picks on validation only; the test score beside each step is report-only.

    uv run python -m megspeech.evaluation.studies.diversity ema-cdrop25-s0 ema-cdrop25-s1 --grouping kaggle
"""

import argparse
import itertools
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.metric.ruler import Graded, Ruler
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Score
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

PARTITIONS = (Partition.VALIDATION, Partition.TEST)
SEED_SUFFIX = re.compile(r"-s\d+$")
LIMIT = 8
"""Models a greedy ensemble grows to."""


class Relation(StrEnum):
    SEED = "seed-seed"
    KNOB = "same-scheme-different-knob"
    SCHEME = "different-scheme"

    @staticmethod
    def of(a: str, b: str) -> "Relation":
        """`a` and `b` differ by seed only, by a knob of one scheme (the leading name token), or by scheme."""
        recipes = SEED_SUFFIX.sub("", a), SEED_SUFFIX.sub("", b)
        if recipes[0] == recipes[1]:
            return Relation.SEED
        return Relation.KNOB if a.split("-")[0] == b.split("-")[0] else Relation.SCHEME


@dataclass(frozen=True)
class Reads:
    """Context-head log-probabilities of the scored words of each counted domain, with their labels."""

    logp: dict[str, Float[Tensor, "n v"]]
    labels: dict[str, Int[Tensor, " n"]]

    @staticmethod
    def mean(members: list["Reads"]) -> "Reads":
        """Members averaged in log-probability space and renormalised, as `Ensemble` and `Evaluator.logits` do."""
        logp = {
            name: torch.stack([member.logp[name] for member in members]).mean(0).log_softmax(-1)
            for name in members[0].logp
        }
        return Reads(logp, members[0].labels)

    def same_words(self, other: "Reads") -> bool:
        return self.labels.keys() == other.labels.keys() and all(
            torch.equal(self.labels[name], other.labels[name]) for name in self.labels
        )

    def truth(self) -> Float[np.ndarray, " n"]:
        """Log-probability of the true class of every scored word, domains pooled."""
        picked = [self.logp[name].gather(1, self.labels[name][:, None])[:, 0] for name in self.logp]
        return torch.cat(picked).cpu().numpy()


@dataclass(frozen=True)
class Scored:
    macro: float
    tau: float


@dataclass(frozen=True)
class Board:
    """The ruler of one partition: scores `Reads` as the selection macro does."""

    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]

    def score(self, reads: Reads, tau: float | None = None) -> Scored:
        """The macro at `tau`, by default the best adjustment of these reads' own macro (the one `ioi` reports)."""
        if tau is None:
            by_domain = [
                self.evaluator.evaluate(reads.logp[name], reads.labels[name], self.log_prior) for name in reads.logp
            ]
            tau = Score.mean(by_domain).tau
        ruler = Ruler()
        graded = Graded(
            {
                name: (
                    ruler.hits(self.evaluator.adjusted(reads.logp[name], self.log_prior, tau), reads.labels[name]),
                    reads.labels[name],
                )
                for name in reads.logp
            }
        )
        return Scored(ruler.point(graded), tau)


@dataclass(frozen=True)
class Member:
    name: str
    reads: dict[Partition, Reads]

    @staticmethod
    def collect(
        root: DataRoot, name: str, scoring: Scoring, device: torch.device
    ) -> tuple["Member", dict[Partition, Board]]:
        """One model read on both partitions: its counted domains' unseen-span words, kept on `device`."""
        loaded = Loader.of(root, [name], scoring, device)
        runs = Run.discover(root.raw())
        reads: dict[Partition, Reads] = {}
        boards: dict[Partition, Board] = {}
        for partition in PARTITIONS:
            held = replace(loaded, partition=partition)
            domains, _ = held.domains(held.ruled(runs))
            counted = {
                domain: split
                for domain, split in domains.items()
                if len(split.novel(held.evaluator.seen)) >= held.evaluator.min_novel
            }
            novel = {domain: split.novel(held.evaluator.seen) for domain, split in counted.items()}
            reads[partition] = Reads(
                {domain: held.context(split)[novel[domain]] for domain, split in counted.items()},
                {domain: split.y[novel[domain]] for domain, split in counted.items()},
            )
            boards[partition] = Board(held.evaluator, held.log_prior)
        return Member(name, reads), boards


@dataclass(frozen=True)
class Step:
    names: tuple[str, ...]
    val: Scored
    test: float


@dataclass(frozen=True)
class Study:
    """Members read on both partitions, and the boards that score them."""

    members: list[Member]
    boards: dict[Partition, Board]

    def ensemble(self, names: list[str] | tuple[str, ...], partition: Partition) -> Reads:
        chosen = [member.reads[partition] for member in self.members if member.name in names]
        return Reads.mean(chosen)

    def single(self, member: Member) -> tuple[Scored, Scored]:
        """Validation score at its own tau, and test score at that tau."""
        val = self.boards[Partition.VALIDATION].score(member.reads[Partition.VALIDATION])
        test = self.boards[Partition.TEST].score(member.reads[Partition.TEST], val.tau)
        return val, test

    def pair_scores(self) -> Float[np.ndarray, "m m"]:
        """Validation macro of every 2-model ensemble; the diagonal is the single model."""
        size = len(self.members)
        out = np.zeros((size, size))
        for i, j in itertools.combinations_with_replacement(range(size), 2):
            names = (self.members[i].name, self.members[j].name)
            reads = self.ensemble(names, Partition.VALIDATION)
            out[i, j] = out[j, i] = self.boards[Partition.VALIDATION].score(reads).macro
        return out

    def correlation(self, partition: Partition) -> Float[np.ndarray, "m m"]:
        """Pearson correlation of the members' true-class log-probabilities."""
        return np.atleast_2d(np.corrcoef(np.stack([member.reads[partition].truth() for member in self.members])))

    def greedy(self, limit: int = LIMIT) -> list[Step]:
        """Forward selection on validation (best single first, then the addition that most raises the validation
        macro, no repeats); each step also reports the test macro at the validation tau, never used to pick."""
        chosen: list[str] = []
        steps: list[Step] = []
        for _ in range(min(limit, len(self.members))):
            options = [member.name for member in self.members if member.name not in chosen]
            scored = {
                name: self.boards[Partition.VALIDATION].score(self.ensemble([*chosen, name], Partition.VALIDATION))
                for name in options
            }
            best = max(scored, key=lambda name: scored[name].macro)
            chosen.append(best)
            test = self.boards[Partition.TEST].score(self.ensemble(chosen, Partition.TEST), scored[best].tau)
            steps.append(Step(tuple(chosen), scored[best], test.macro))
        return steps

    def groups(self, matrix: Float[np.ndarray, "m m"]) -> dict[Relation, list[float]]:
        """Each off-diagonal pair's entry of `matrix`, by how its two models' recipes relate."""
        grouped: dict[Relation, list[float]] = defaultdict(list)
        for i, j in itertools.combinations(range(len(self.members)), 2):
            grouped[Relation.of(self.members[i].name, self.members[j].name)].append(float(matrix[i, j]))
        return grouped

    def summary(self, matrix: Float[np.ndarray, "m m"]) -> str:
        return " | ".join(
            f"{relation} n={len(values)} mean {np.mean(values):.4f} min {np.min(values):.4f} max {np.max(values):.4f}"
            for relation, values in self.groups(matrix).items()
        )

    def table(self, matrix: Float[np.ndarray, "m m"]) -> list[str]:
        """Rows of `matrix` labelled by member index."""
        return [f"{i:2d} " + " ".join(f"{value:.3f}" for value in row) for i, row in enumerate(matrix)]

    def log(self) -> None:
        for index, member in enumerate(self.members):
            val, test = self.single(member)
            LOG.info("single %2d %s val %.4f tau %s test %.4f", index, member.name, val.macro, val.tau, test.macro)
        pairs = self.pair_scores()
        for partition in PARTITIONS:
            matrix = self.correlation(partition)
            LOG.info("true-class log-prob correlation %s: %s", partition, self.summary(matrix))
            LOG.info("correlation matrix %s\n%s", partition, "\n".join(self.table(matrix)))
        LOG.info("pair ensemble val macro, grouped: %s", self.summary(pairs))
        LOG.info("pair ensemble val macro matrix (diagonal = single)\n%s", "\n".join(self.table(pairs)))
        for size, step in enumerate(self.greedy(), 1):
            LOG.info(
                "greedy %d +%s val %.4f tau %s test %.4f", size, step.names[-1], step.val.macro, step.val.tau, step.test
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    tree = Tree(Scoring, Scoring(fallback=Fallback(Unseen.DEEP)))
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, scoring, device = DataRoot.from_config(), tree.parse(args), Device.CUDA.as_torch()
    members: list[Member] = []
    boards: dict[Partition, Board] = {}
    for name in args.names:
        try:
            member, own = Member.collect(root, name, scoring, device)
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            LOG.warning("skipped %s: %s: %s", name, type(error).__name__, error)
            continue
        if members and not all(member.reads[p].same_words(members[0].reads[p]) for p in PARTITIONS):
            LOG.warning("skipped %s: scored words differ from %s", name, members[0].name)
            continue
        members.append(member)
        boards = boards or own
        torch.cuda.empty_cache()
    Study(members, boards).log()


if __name__ == "__main__":
    main()

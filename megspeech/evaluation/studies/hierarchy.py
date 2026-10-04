"""Whether a part-of-speech posterior, trust-weighted, improves the 50-way decoder when late-fused into its logits
(megspeech-7gf.94): every scored word's vocabulary scores plus `lam` times log sum_t p(t | MEG) p(t | word), the stacked
probe's calibrated posterior over each word's tag against the tag distribution each vocabulary word shows in context on
the training runs. The model is frozen and the fusion runs on the CPU.

`lam` is fit leave-one-domain-out over the official validation domains: chosen on two, scored on the third, the macro
read as the official scorer reads it. The score at every grid `lam` over all domains is a diagnostic fit on validation.

    uv run python -m megspeech.evaluation.studies.hierarchy sent40-ioi-ema9998-s0
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.labels.syntax import PartsOfSpeech
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import DEEP, Run
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head, Score, SelectionConfig
from megspeech.evaluation.studies.syntax import FLOOR, Embedded, Probe, Table
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.subjects import Fallback, SubjectLayer, Unseen
from megspeech.model.sentence import SentenceNet

LOG = logging.getLogger(__name__)

GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
"""Fusion weights on the part-of-speech bonus; 0 is the context head alone."""
STACK = ("timing", "meg-context")
"""The probes whose out-of-fold log-probabilities the stacker reads."""
OFFICIAL = 0.6445
"""The context head's official macro BAcc@10 for sent40-ioi-ema9998-s0 (n=2512): the plumbing check's target."""
TOLERANCE = 0.002
"""How far the un-fused macro may sit from `OFFICIAL` before the plumbing is distrusted."""


@dataclass(frozen=True)
class WordTags:
    """Each vocabulary word's distribution over the stacker's tags, counted in context on training rows, add-one."""

    distribution: Float[np.ndarray, "v k"]

    @classmethod
    def of(cls, targets: Int[np.ndarray, " n"], tags: Int[np.ndarray, " n"], k: int) -> "WordTags":
        """The distribution of `tags` (each below `k`) over the rows whose 50-way `targets` name a word."""
        counts = np.ones((VOCAB_SIZE, k))
        inside = targets >= 0
        np.add.at(counts, (targets[inside], tags[inside]), 1)
        return cls(counts / counts.sum(1, keepdims=True))

    def bonus(self, posterior: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n v"]:
        """Each row's log-likelihood, under every word's tags, of the tag posterior's mass."""
        return np.log(np.exp(posterior) @ self.distribution.T + FLOOR)


@dataclass(frozen=True)
class Domain:
    """One validation domain's unseen-span words: the vocabulary scores, each word's part-of-speech bonus, its label."""

    logits: Float[Tensor, "n v"]
    bonus: Float[Tensor, "n v"]
    labels: Int[Tensor, " n"]

    @classmethod
    def of(  # devtools-ignore: test-mirror
        cls,
        logits: Float[Tensor, "n v"],
        labels: Int[Tensor, " n"],
        words: pd.DataFrame,
        valid: Table,
        bonus: Float[np.ndarray, "m v"],
    ) -> "Domain":
        """`logits` and `labels` of `words`' rows, each given its bonus from the validation table row of the same run
        and onset; a word the table dropped for want of a tag has none."""
        rows = valid.rows(words["run"].to_numpy(), words["onset"].to_numpy(np.float64))
        found = rows >= 0
        wanted = labels.cpu().numpy()
        if (valid.targets[rows[found]] != wanted[found]).any():
            raise ValueError("a scored word and its part-of-speech row name different vocabulary words")
        aligned = np.zeros((len(rows), bonus.shape[1]))
        aligned[found] = bonus[rows[found]]
        LOG.info("%d of %d scored words carry a part of speech", found.sum(), len(rows))
        return cls(logits, torch.from_numpy(aligned).float().to(logits.device), labels)

    def fused(self, lam: float) -> Float[Tensor, "n v"]:
        """The vocabulary scores with `lam` times the bonus added."""
        return self.logits + lam * self.bonus


@dataclass(frozen=True)
class Grid:
    """Every domain's score at every `lam`, each the official ruler's per-adjustment balanced accuracy."""

    scores: dict[str, dict[float, Score]]

    @classmethod
    def of(cls, evaluator: Evaluator, log_prior: Float[Tensor, " v"], domains: dict[str, Domain]) -> "Grid":
        """`domains` scored at every `GRID` weight."""
        return cls(
            {
                name: {lam: evaluator.evaluate(domain.fused(lam), domain.labels, log_prior) for lam in GRID}
                for name, domain in domains.items()
            }
        )

    def macro(self, lam: float) -> Score:
        """The macro over every domain at `lam`, the adjustment chosen once for all as the official scorer does."""
        return Score.mean(by_lam[lam] for by_lam in self.scores.values())

    def chosen(self, held: str) -> float:
        """The weight that scores best on every domain but `held`, the smaller on a tie."""
        others = [by_lam for name, by_lam in self.scores.items() if name != held]
        return max(GRID, key=lambda lam: Score.mean(by_lam[lam] for by_lam in others).bacc10)

    def leave_one_out(self) -> tuple[Score, dict[str, float]]:
        """The macro of each domain scored at the weight the others chose, and those weights."""
        weights = {name: self.chosen(name) for name in self.scores}
        return Score.mean(self.scores[name][lam] for name, lam in weights.items()), weights

    def per_domain(self, weights: dict[str, float], macro: Score) -> dict[str, tuple[float, int]]:
        """Each domain's score at its weight, read at the macro's adjustment, and its word count."""
        return {
            name: (round(self.scores[name][lam].by_tau[macro.tau], 4), self.scores[name][lam].n)
            for name, lam in weights.items()
        }

    def report(self) -> None:  # devtools-ignore: test-mirror
        """Log the plumbing check, the fit-on-validation grid and the leave-one-domain-out macro."""
        base = self.macro(0.0)
        LOG.info("lam 0 macro %.4f over %d words, official %.4f", base.bacc10, base.n, OFFICIAL)
        if abs(base.bacc10 - OFFICIAL) > TOLERANCE:
            raise ValueError(f"un-fused macro {base.bacc10:.4f} does not reproduce the official {OFFICIAL}")
        for lam in GRID:
            macro = self.macro(lam)
            LOG.info(
                "fit-on-val lam %s macro %.4f tau %s over %d words, per domain %s",
                lam,
                macro.bacc10,
                macro.tau,
                macro.n,
                self.per_domain(dict.fromkeys(self.scores, lam), macro),
            )
        macro, weights = self.leave_one_out()
        LOG.info(
            "leave-one-domain-out macro %.4f tau %s over %d words, per domain %s, lam %s",
            macro.bacc10,
            macro.tau,
            macro.n,
            self.per_domain(weights, macro),
            weights,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained model under models/")
    parser.add_argument("--stride", type=int, default=4, help="every stride-th subject-0 training run is probed on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    directories = Ensemble.trained(root.models(), args.name)
    data = DataConfig.shared(directories, DataConfig.grouping)
    reader = Reader(root, data, device, geometry=data.geometry(root, device, aligned=False), official=True)
    model = Ensemble.load([directory / "model.pt" for directory in directories], Unseen.DEEP).to(device)
    for layer in (module for module in model.modules() if isinstance(module, SubjectLayer)):
        layer.shrink(Fallback(Unseen.DEEP).keep, Unseen.DEEP)
    sequences, log_prior = reader.heard()
    evaluator = Evaluator(device, sequences, SelectionConfig.shared(directories).min_novel)
    runs = Run.discover(root.raw())
    held = reader.ruled(runs)
    trained = [run for run in data.trained(runs) if run.subject == DEEP][:: args.stride]
    parts = PartsOfSpeech.load(PartsOfSpeech.path(root))
    embedded = Embedded(SentenceNet.load(root.models() / args.name / "model.pt").to(device))
    train, valid = Table.of(reader, embedded, {"train": trained}, parts), Table.of(reader, embedded, held, parts)
    probe = Probe(train, valid)
    classes, posterior = probe.stacked(STACK, {name: probe.probabilities(name) for name in STACK})
    bonus = WordTags.of(train.targets, np.searchsorted(classes, train.tags), len(classes)).bonus(posterior)
    splits, _ = reader.domains(held)
    official = evaluator.report(model, splits, log_prior, 0, 0.0)
    LOG.info("official macro %.4f over %d words, expected %.4f", official.macro.bacc10, official.macro.n, OFFICIAL)
    _, words = reader.load([run for group in held.values() for run in group])
    domains = {}
    for name, split in splits.items():
        novel = split.novel(evaluator.seen)
        if len(novel) >= evaluator.min_novel:
            rows = words[words["run"].isin([run.name for run in held[name]])].iloc[novel.cpu().numpy()]
            logits = evaluator.logits(model, split)[Head.CONTEXT][novel]
            domains[name] = Domain.of(logits, split.y[novel], rows, valid, bonus)
    Grid.of(evaluator, log_prior, domains).report()


if __name__ == "__main__":
    main()

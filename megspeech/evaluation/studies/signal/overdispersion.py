"""Whether word-decoding correctness clusters by sentence beyond what per-word difficulty predicts.

Each scored word gets a score `s` (the percentile rank of its true label among the vocabulary under the saved
probabilities, 1 = top) and a hit `h` (the label is in the top `TOP`). Per-word difficulty is removed by the
leave-one-out residual: the word's value minus the mean of the OTHER words of its stratum (the domain's label). The
statistic is the one-way intraclass correlation (ICC1) of the residual over spans, the between-span share of its
variance. Its null permutes the spans among the words of each stratum, so label difficulty and span sizes are kept and
only the word-to-span assignment is broken.

A span is a sentence as the model's grouping cuts it: a sentence longer than nine words is several spans. Words come
from the unseen-span (`novel`) words of `probs.pt`, the ones every score counts. The pooled row permutes within domain
and label, never across domains, so a domain's own difficulty is not read as a sentence effect.

    uv run python -m megspeech.evaluation.studies.signal.overdispersion <run> --section local-test --perms 2000
"""

import argparse
import logging
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.probabilities import PROBS, SECTIONS, Probabilities, Words
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

TOP = 10
"""A hit is the true label among this many of the vocabulary's best."""
LOW, HIGH = 0.025, 0.975
"""The null interval's quantiles: its central 95%."""
CHUNK = 250
"""Permutations drawn at once, bounding the `perms x words` index tensor."""
POOLED = "pooled"
VAL = "val"
CPU = Device.CPU.as_torch()
ABSENT = -1
"""A word no span reads."""


@dataclass(frozen=True)
class Outcome:
    """A statistic's observed value against its permutation null."""

    observed: float
    null_mean: float
    low: float
    high: float
    p: float
    words: int
    spans: int

    def line(self, name: str, measure: str) -> str:
        return (
            f"{name:>14} {measure:>5} n_words={self.words} n_spans={self.spans} icc={self.observed:+.4f} "
            f"null={self.null_mean:+.4f} [{self.low:+.4f}, {self.high:+.4f}] p={self.p:.4f}"
        )


@dataclass(frozen=True)
class Ranks:
    """Each word's true label scored against the vocabulary: `score` its percentile rank (1 = top), `hit` whether it is
    among the best `TOP`."""

    score: Float[Tensor, " n"]
    hit: Float[Tensor, " n"]

    @classmethod
    def of(cls, probabilities: Float[Tensor, "n v"], labels: Int[Tensor, " n"]) -> "Ranks":
        true = probabilities.gather(1, labels[:, None])
        beaten_by = (probabilities > true).sum(dim=1)
        return cls(1 - beaten_by / (probabilities.shape[1] - 1), (beaten_by < TOP).float())


@dataclass(frozen=True)
class Cases:
    """Scored words, each in a `stratum` (what sets its difficulty) and a `span`; difficulty-free residuals of `values`
    are what clusters."""

    stratum: Int[Tensor, " n"]
    span: Int[Tensor, " n"]
    ranks: Ranks

    @staticmethod
    def residual(values: Float[Tensor, " n"], stratum: Int[Tensor, " n"]) -> Float[Tensor, " n"]:
        """`values` minus the mean of the other words in each stratum; a lone word of its stratum has no other, NaN."""
        _, group = stratum.unique(return_inverse=True)
        total = torch.zeros(int(group.max()) + 1).scatter_add_(0, group, values)
        count = torch.zeros_like(total).scatter_add_(0, group, torch.ones_like(values))
        others = (total[group] - values) / (count[group] - 1)
        return values - others

    def measure(self, name: str) -> Float[Tensor, " n"]:
        return {"score": self.ranks.score, "hit": self.ranks.hit}[name]

    def dispersion(self, name: str) -> "Dispersion":
        """The residuals of measure `name` over the words that have a stratum-mate and a span, ordered by stratum."""
        residual = self.residual(self.measure(name), self.stratum)
        keep = residual.isfinite() & (self.span != ABSENT)
        order = self.stratum[keep].argsort(stable=True)
        _, span = self.span[keep][order].unique(return_inverse=True)
        return Dispersion(residual[keep][order], self.stratum[keep][order], span)


@dataclass(frozen=True)
class Dispersion:
    """Residuals with the stratum and dense span of each word, ordered so a stratum's words are contiguous."""

    residual: Float[Tensor, " n"]
    stratum: Int[Tensor, " n"]
    span: Int[Tensor, " n"]

    @property
    def spans(self) -> int:
        return int(self.span.max()) + 1 if len(self.span) else 0

    def icc(self, assignment: Int[Tensor, "p n"]) -> Float[Tensor, " p"]:
        """The one-way ICC1 of the residuals for each row of `assignment`, a span index per word."""
        words, groups = self.residual.shape[0], self.spans
        sizes = torch.bincount(self.span, minlength=groups).float()
        sums = torch.zeros(len(assignment), groups).scatter_add_(
            1, assignment, self.residual.expand(len(assignment), -1)
        )
        mean = self.residual.mean()
        between = (sums.square() / sizes).sum(dim=1) - words * mean.square()
        within = self.residual.square().sum() - (sums.square() / sizes).sum(dim=1)
        msb, msw = between / (groups - 1), within / (words - groups)
        size = (words - sizes.square().sum() / words) / (groups - 1)
        return (msb - msw) / (msb + (size - 1) * msw)

    def shuffled(self, count: int, generator: torch.Generator) -> Int[Tensor, "p n"]:
        """`count` assignments of the spans to words, each permuting them among the words of every stratum."""
        _, stratum = self.stratum.unique(return_inverse=True)
        keys = stratum[None, :] + torch.rand(count, len(stratum), generator=generator)
        return self.span[keys.argsort(dim=1)]

    def test(self, perms: int, seed: int = 0) -> Outcome:
        """The observed ICC, the permutation null's mean and central 95%, and the one-sided p of a null at least as
        clustered."""
        generator = torch.Generator().manual_seed(seed)
        observed = float(self.icc(self.span[None, :])[0])
        null = torch.cat(
            [self.icc(self.shuffled(min(CHUNK, perms - done), generator)) for done in range(0, perms, CHUNK)]
        )
        edges = null.quantile(torch.tensor([LOW, HIGH]))
        p = (1 + float((null >= observed).sum())) / (1 + perms)
        return Outcome(observed, float(null.mean()), float(edges[0]), float(edges[1]), p, len(self.span), self.spans)


@dataclass(frozen=True)
class Spans:
    """Which span of a split reads each of its words."""

    of_word: Int[Tensor, " n"]

    @classmethod
    def of(cls, split: Split) -> "Spans":
        rows, columns = torch.nonzero(split.index != PAD, as_tuple=True)
        of_word = torch.full((len(split.y),), ABSENT, dtype=torch.long)
        of_word[split.index[rows, columns].cpu()] = rows.cpu()
        return cls(of_word)

    def read(self, ids: Int[Tensor, " k"]) -> Int[Tensor, " k"]:
        return self.of_word[ids]


@dataclass(frozen=True)
class Overdispersion:
    """A section's scored words by domain, as `Cases`, tested per domain and pooled."""

    domains: dict[str, Cases]

    @classmethod
    def of(cls, saved: dict[str, Words], splits: dict[str, Split]) -> "Overdispersion":
        """The `novel` words of every domain `saved` and `splits` share."""
        cases = {}
        for name in saved.keys() & splits.keys():
            words = saved[name]
            novel: Bool[Tensor, " n"] = words.novel
            span = Spans.of(splits[name]).read(words.ids[novel])
            cases[name] = Cases(words.labels[novel], span, Ranks.of(words.probabilities[novel], words.labels[novel]))
        return cls(dict(sorted(cases.items())))

    def pooled(self) -> Cases:
        """Every domain's words, strata and spans kept apart by domain."""
        parts = list(self.domains.values())
        width = 1 + max(int(part.span.max()) for part in parts)
        spans = [torch.where(part.span == ABSENT, ABSENT, part.span + i * width) for i, part in enumerate(parts)]
        strata = [part.stratum + i * VOCAB_SIZE for i, part in enumerate(parts)]
        score = torch.cat([part.ranks.score for part in parts])
        hit = torch.cat([part.ranks.hit for part in parts])
        return Cases(torch.cat(strata), torch.cat(spans), Ranks(score, hit))

    def report(self, perms: int, seed: int = 0) -> list[str]:
        """One line per domain, then pooled, for the score and the hit."""
        rows = {**self.domains, POOLED: self.pooled()}
        return [
            cases.dispersion(measure).test(perms, seed).line(name, measure)
            for name, cases in rows.items()
            for measure in ("score", "hit")
        ]


class Study:
    @staticmethod
    def _splits(root: DataRoot, name: str, section: str) -> dict[str, Split]:
        """The splits `section` was scored on, rebuilt from the model's own recorded data config."""
        directories = Ensemble.trained(root.models(), name)
        data = DataConfig.shared(directories, DataConfig.saved(directories[0]).grouping)
        if section == VAL:
            return Corpus.load(data, root, CPU).validation
        partition = next(part for part, saved in SECTIONS.items() if saved == section)
        reader = Reader(root, data, CPU, partition=partition)
        return reader.domains(reader.ruled(Run.discover(root.raw())))[0]

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="a trained run under models/ with a probs.pt")
        parser.add_argument("--section", default="local-test", choices=[VAL, *SECTIONS.values()])
        parser.add_argument("--perms", type=int, default=2000)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        saved = Probabilities.load(root.models() / args.name / PROBS, args.section)
        study = Overdispersion.of(saved.domains, Study._splits(root, args.name, args.section))
        for line in study.report(args.perms, args.seed):
            LOG.info(line)


if __name__ == "__main__":
    Study.main()

"""The selection ruler on the device: balanced top-k accuracy, its per-class coverage, and a bootstrap interval.

Validation holds a median of 4-13 words per class, so a balanced accuracy moves by whole classes; the
interval resamples words with replacement and re-scores, all resamples at once. A fixed seed draws the
same resamples for every run scored on the same words, so differences between runs are paired:
`python -m megspeech.evaluation.ruler A/hits.pt B/hits.pt` bounds how much A beats B by.
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from megspeech.data.vocabulary import WORDS

LOG = logging.getLogger(__name__)

TOP = 10
RESAMPLES = 1000
CONFIDENCE = 0.95
SPARSE = 5
"""Classes with fewer words than this score in steps of 20% or more."""


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class Coverage:
    """How many classes a subset holds, the median words per held class, and how many hold fewer than `SPARSE`."""

    present: int
    median: float
    sparse: int

    @classmethod
    def of(cls, labels: Int[Tensor, " n"]) -> "Coverage":
        counts = torch.bincount(labels, minlength=WORDS)
        held = counts[counts > 0]
        if not len(held):
            return cls(0, 0.0, 0)
        return cls(len(held), float(held.float().median()), int((held < SPARSE).sum()))


@dataclass(frozen=True)
class Graded:
    """One model's per-word hits and labels on each selection domain, kept beside the model for paired comparison."""

    domains: dict[str, tuple[Bool[Tensor, " n"], Int[Tensor, " n"]]]

    def save(self, path: Path) -> Path:
        torch.save({name: (hits.cpu(), labels.cpu()) for name, (hits, labels) in self.domains.items()}, path)
        return path

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "Graded":
        stored = torch.load(path, map_location=device, weights_only=True)
        return cls({name: (hits, labels) for name, (hits, labels) in stored.items()})


@dataclass(frozen=True)
class Ruler:
    k: int = TOP
    resamples: int = RESAMPLES
    confidence: float = CONFIDENCE
    seed: int = 0

    def hits(self, scores: Float[Tensor, "n v"], labels: Int[Tensor, " n"]) -> Bool[Tensor, " n"]:
        """Whether the true word ranks in the top `k` (ties counted as hits, a non-finite row as a miss)."""
        true = scores.gather(1, labels[:, None])
        return ((scores > true).sum(dim=1) < self.k) & scores.isfinite().all(dim=1)

    def balanced(self, hits: Bool[Tensor, " n"], labels: Int[Tensor, " n"]) -> float:
        """Mean over present classes of the hit rate."""
        return float(self.score(hits[None], labels[None])[0])

    def draws(self, hits: Bool[Tensor, " n"], labels: Int[Tensor, " n"]) -> Float[Tensor, " b"]:
        """Balanced accuracy of each resample of the words, drawn with replacement."""
        generator = torch.Generator(device=labels.device).manual_seed(self.seed)
        pick = torch.randint(len(labels), (self.resamples, len(labels)), generator=generator, device=labels.device)
        return self.score(hits[pick], labels[pick])

    def point(self, graded: Graded) -> float:
        """Mean over domains of their balanced accuracy, the selection macro itself."""
        return float(np.mean([self.balanced(hits, labels) for hits, labels in graded.domains.values()]))

    def macro(self, graded: Graded) -> Float[Tensor, " b"]:
        """Resampled mean over domains, each domain resampled within itself; NaN draws when no domain counts."""
        if not graded.domains:
            return torch.full((self.resamples,), math.nan)
        return torch.stack([self.draws(hits, labels) for hits, labels in graded.domains.values()]).mean(dim=0)

    def delta(self, a: Graded, b: Graded) -> Float[Tensor, " b"]:
        """Paired resampled macro difference a - b; both must be graded on the same words."""
        same = a.domains.keys() == b.domains.keys() and all(
            torch.equal(a.domains[name][1], b.domains[name][1]) for name in a.domains
        )
        if not same:
            raise ValueError("runs were graded on different words; a paired difference needs the same ones")
        return self.macro(a) - self.macro(b)

    def interval(self, draws: Float[Tensor, " b"]) -> Interval:
        tail = (1 - self.confidence) / 2
        low, high = torch.quantile(draws, torch.tensor([tail, 1 - tail], device=draws.device)).tolist()
        return Interval(low, high)

    @staticmethod
    def information(predicted: Int[Tensor, " n"], labels: Int[Tensor, " n"], prior: Float[Tensor, " v"]) -> float:
        """Plug-in mutual information in bits between the word and the top-1 prediction, each present class's row
        of the confusion matrix weighted by `prior` renormalised over the present classes — the competition's OVMI
        reads it under SUBTLEX-UK frequencies, times a coverage every team shares. 0 when the prediction ignores
        the word; biased upward at a few words per class, so runs compare only on the same words."""
        joint = torch.zeros(WORDS, WORDS, device=labels.device)
        joint.index_put_((labels, predicted), torch.ones_like(labels, dtype=joint.dtype), accumulate=True)
        rows = joint.sum(dim=1, keepdim=True)
        weight = torch.where(rows > 0, prior[:, None], 0.0)
        joint = joint / rows.clamp(min=1) * weight / weight.sum()
        expected = joint.sum(dim=1, keepdim=True) * joint.sum(dim=0, keepdim=True)
        return float(torch.where(joint > 0, joint * (joint / expected).log2(), 0.0).sum())

    @staticmethod
    def score(hits: Bool[Tensor, "b n"], labels: Int[Tensor, "b n"]) -> Float[Tensor, " b"]:
        """Balanced accuracy of every row, one bincount over rows offset into their own class block."""
        rows = len(labels)
        slot = (labels + WORDS * torch.arange(rows, device=labels.device)[:, None]).flatten()
        counts = torch.bincount(slot, minlength=rows * WORDS).view(rows, WORDS)
        correct = torch.bincount(slot, weights=hits.flatten().float(), minlength=rows * WORDS).view(rows, WORDS)
        return (correct / counts.clamp(min=1)).sum(dim=1) / (counts > 0).sum(dim=1)


def main() -> None:
    """Log how much run A beats run B on the selection macro, with its paired bootstrap interval."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("a", type=Path, help="hits.pt of run A")
    parser.add_argument("b", type=Path, help="hits.pt of run B")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ruler = Ruler()
    a, b = Graded.load(args.a, device), Graded.load(args.b, device)
    delta = ruler.delta(a, b)
    interval = ruler.interval(delta)
    LOG.info(
        "a %.4f  b %.4f  a-b %+.4f  %d%% CI [%+.4f, %+.4f]  P(a>b) %.3f",
        ruler.point(a),
        ruler.point(b),
        float(delta.mean()),
        round(100 * ruler.confidence),
        interval.low,
        interval.high,
        float((delta > 0).float().mean()),
    )


if __name__ == "__main__":
    main()

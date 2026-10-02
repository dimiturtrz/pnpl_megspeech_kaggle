"""The selection ruler on the device: balanced top-k accuracy, its per-class coverage, and a bootstrap interval.

Validation holds a median of 4-13 words per class, so a balanced accuracy moves by whole classes; the
interval resamples words with replacement and re-scores, all resamples at once. A fixed seed draws the
same resamples for every run scored on the same words, so differences between runs are paired.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from megspeech.score import TOP
from megspeech.wordnet import WORDS

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

    def interval(self, draws: Float[Tensor, " b"]) -> Interval:
        tail = (1 - self.confidence) / 2
        low, high = torch.quantile(draws, torch.tensor([tail, 1 - tail], device=draws.device)).tolist()
        return Interval(low, high)

    @staticmethod
    def score(hits: Bool[Tensor, "b n"], labels: Int[Tensor, "b n"]) -> Float[Tensor, " b"]:
        """Balanced accuracy of every row, one bincount over rows offset into their own class block."""
        rows = len(labels)
        slot = (labels + WORDS * torch.arange(rows, device=labels.device)[:, None]).flatten()
        counts = torch.bincount(slot, minlength=rows * WORDS).view(rows, WORDS)
        correct = torch.bincount(slot, weights=hits.flatten().float(), minlength=rows * WORDS).view(rows, WORDS)
        return (correct / counts.clamp(min=1)).sum(dim=1) / (counts > 0).sum(dim=1)

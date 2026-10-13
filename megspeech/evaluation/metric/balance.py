"""Transductive reshaping of a scored set before its top-k is read: class balancing, and repeat smoothing.

The metric averages recall over classes, and logit adjustment removes the training prior, but a recording can still
under-predict a class (session shift). Sinkhorn-Knopp on the columns shifts each class's scores until every class
holds an equal share of the set's posterior mass; a row's ranks are all top-k reads, so rows need no normalising.

Rows of one scored set that are the same word should agree (megspeech-d6r.8, interpretations/deep/
2026-10-13_pool_smoothing.md), so `PoolSmoothing` mixes each row's probabilities with those of its `k` nearest other
rows, nearness being the cosine between the rows' own probability vectors centred on the set's mean. The setting chosen
on val pools (k 5, temperature .05, lam .1) read +.0064 on the local test for one model.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

FLOOR = 1e-6
QUERY_CHUNK = 1024


@dataclass(frozen=True)
class Sinkhorn:
    """A per-class shift, `steps` Sinkhorn iterations toward the target posterior mass per class; zero steps is none."""

    steps: int = 0
    toward: float = 0.0
    """Exponent on the training prior the class masses are balanced toward, `softmax(toward * log_prior)`: 0 is equal
    mass, 1 the prior itself."""

    per_subject: bool = False
    """Balance each subject's rows on their own, as the Broad ruler does its runs, instead of the whole set at once;
    needs the `groups` a caller passes (the Kaggle test's rows' subjects)."""

    def __call__(
        self,
        scores: Float[Tensor, "n v"],
        log_prior: Float[Tensor, " v"],
        groups: Int[Tensor, " n"] | None = None,
    ) -> Float[Tensor, "n v"]:
        balanced = scores.clone()
        for rows in self._parts(groups):
            balanced[rows] = self.shift(scores[rows], log_prior)
        return balanced

    def _parts(self, groups: Int[Tensor, " n"] | None) -> list[Int[Tensor, " n"] | slice]:
        """The row selectors balanced separately: every row together, or one per distinct value of `groups`."""
        if not self.per_subject or groups is None:
            return [slice(None)]
        return [groups == group for group in groups.unique()]

    def shift(self, scores: Float[Tensor, "n v"], log_prior: Float[Tensor, " v"]) -> Float[Tensor, "n v"]:
        target = (self.toward * log_prior).log_softmax(-1)
        shifted = scores
        for _ in range(self.steps):
            gap = shifted.log_softmax(-1).logsumexp(0) - target
            shifted = shifted - (gap - gap.mean())
        return shifted


@dataclass(frozen=True)
class PoolSmoothing:
    """Each row's probabilities mixed at `lam` with its `k` nearest other rows', weighted `softmax(cos / temperature)`;
    zero `k` is none. Rows of different `groups` (the Kaggle test's subjects) never neighbour."""

    k: int = 0
    temperature: float = 0.05
    lam: float = 0.1

    def __call__(self, logits: Float[Tensor, "n v"], groups: Int[Tensor, " n"] | None = None) -> Float[Tensor, "n v"]:
        if not self.k:
            return logits
        if groups is None:
            return self.mixed(logits)
        smoothed = logits.clone()
        for group in groups.unique():
            rows = groups == group
            smoothed[rows] = self.mixed(logits[rows])
        return smoothed

    def mixed(self, logits: Float[Tensor, "n v"]) -> Float[Tensor, "n v"]:
        """One set's log-probabilities after the mix; a set of one row has no neighbour and is returned normalised."""
        log_p = logits.log_softmax(-1)
        p = log_p.exp()
        depth = min(self.k, len(p) - 1)
        if depth < 1:
            return log_p
        unit = functional.normalize(p - p.mean(dim=0), dim=-1)
        readout = torch.cat([self.readout(unit, p, start, depth) for start in range(0, len(p), QUERY_CHUNK)])
        return torch.log((1 - self.lam) * p + self.lam * readout.clamp(min=FLOOR))

    def readout(
        self, unit: Float[Tensor, "n v"], p: Float[Tensor, "n v"], start: int, depth: int
    ) -> Float[Tensor, "c v"]:
        """The weighted neighbours' probabilities for the query rows from `start`, each row excluded from its own."""
        stop = min(start + QUERY_CHUNK, len(unit))
        sims = unit[start:stop] @ unit.T
        sims[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        near, at = sims.topk(depth, dim=-1)
        weights = (near / self.temperature).softmax(dim=-1)
        return torch.einsum("ck,ckv->cv", weights, p[at])

"""Transductive class balancing of a scored set before its top-k is read.

The metric averages recall over classes, and logit adjustment removes the training prior, but a recording can still
under-predict a class (session shift). Sinkhorn-Knopp on the columns shifts each class's scores until every class
holds an equal share of the set's posterior mass; a row's ranks are all top-k reads, so rows need no normalising.
"""

from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor


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

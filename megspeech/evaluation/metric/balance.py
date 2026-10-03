"""Transductive class balancing of a scored set before its top-k is read.

The metric averages recall over classes, and logit adjustment removes the training prior, but a recording can still
under-predict a class (session shift). Sinkhorn-Knopp on the columns shifts each class's scores until every class
holds an equal share of the set's posterior mass; a row's ranks are all top-k reads, so rows need no normalising.
"""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class Sinkhorn:
    """A per-class shift, `steps` Sinkhorn iterations toward equal posterior mass per class; zero steps is none."""

    steps: int = 0

    def __call__(self, scores: Float[Tensor, "n v"]) -> Float[Tensor, "n v"]:
        shifted = scores
        for _ in range(self.steps):
            mass = shifted.log_softmax(-1).logsumexp(0)
            shifted = shifted - (mass - mass.mean())
        return shifted

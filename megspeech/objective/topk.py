"""Smooth top-k SVM loss (Berrada, Zisserman & Kumar 2018, "Smooth Loss Functions for Deep Top-k Classification").

The metric counts a word as hit when it ranks in the top 10; cross-entropy keeps pushing a hit word towards rank 1.
This loss is satisfied once the label leads every k-subset without it by the margin, smoothed by temperature tau.
Its log-sum-exp over k-subsets is a ratio of elementary symmetric polynomials of exp(s / (k tau)), computed exactly
by the recursion e_j <- e_j + x_i e_{j-1} in log space.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE

TOP = 10
"""The metric's k: BAcc@10."""
ABSENT = -1e4
"""log of a term that contributes nothing: finite, so log-sum-exp gradients stay 0 rather than NaN."""


@dataclass(frozen=True)
class SmoothTopK:
    k: int = TOP
    margin: float = 1.0
    tau: float = 1.0

    def __call__(
        self, logits: Float[Tensor, "n v"], y: Int[Tensor, " n"], weight: Float[Tensor, " v"] | None = None
    ) -> Float[Tensor, ""]:
        """Mean over the rows not labelled `IGNORE`, each weighted by its class's `weight` as cross-entropy weighs."""
        keep = y != IGNORE
        label = y.clamp(min=0)
        scaled = logits.float() / (self.k * self.tau)
        own = scaled.gather(-1, label[:, None])[:, 0]
        others = scaled.scatter(-1, label[:, None], ABSENT)
        symmetric = self.symmetric(others)
        without, beside = symmetric[:, self.k], symmetric[:, self.k - 1] + own
        rows = self.tau * functional.softplus(self.margin / self.tau + without - beside)
        scale = keep.float() if weight is None else keep.float() * weight[label]
        return (rows * scale).sum() / scale.sum()

    def symmetric(self, logs: Float[Tensor, "n v"]) -> Float[Tensor, "n k1"]:
        """log e_j(exp(logs)) for j = 0..k, row by row."""
        floor = torch.full((logs.shape[0], self.k), ABSENT, device=logs.device)
        table = torch.cat([torch.zeros_like(floor[:, :1]), floor], dim=-1)
        for column in logs.unbind(-1):
            grown = torch.logaddexp(table[:, 1:], table[:, :-1] + column[:, None])
            table = torch.cat([table[:, :1], grown], dim=-1)
        return table

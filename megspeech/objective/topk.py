"""Smooth top-k SVM loss (Berrada, Zisserman & Kumar 2018, "Smooth Loss Functions for Deep Top-k Classification").

The metric counts a word as hit when it ranks in the top 10; cross-entropy keeps pushing a hit word towards rank 1.
This loss is satisfied once the label leads every k-subset without it by the margin, smoothed by temperature tau.
Its log-sum-exp over k-subsets is a ratio of elementary symmetric polynomials of exp(s / (k tau)), computed exactly
by the recursion e_j <- e_j + x_i e_{j-1} in log space.

`TopKMix` blends it into a head's own loss in raw-logit units: k times the loss at temperature t/k and margin m/k is
t softplus(m/t + LSE_without(sum s/t) - LSE_with(sum s/t)) over k-subset logit sums, which at k=1, t=1, m=0 is the
cross-entropy itself. Its gradient per raw logit then matches the cross-entropy's scale, and t=1 smooths on the logit
scale the cross-entropy trained; the paper's tau=1 at k=10 is t=10, a margin of 10 raw logits (megspeech-7gf.73).
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


@dataclass(frozen=True)
class TopKMix:
    """A head's loss as `(1 - share) * own + share * k * SmoothTopK`, the latter at raw-logit `temperature` and
    `margin` over the same logits and class weights; `share` 0 is the head's own loss untouched."""

    share: float = 0.0
    temperature: float = 1.0
    margin: float = 1.0
    k: int = TOP

    @property
    def loss(self) -> SmoothTopK:
        """The paper's loss whose k-fold equals this mix's raw-unit term."""
        return SmoothTopK(k=self.k, margin=self.margin / self.k, tau=self.temperature / self.k)

    def mixed(
        self,
        own: Float[Tensor, ""],
        logits: Float[Tensor, "n v"],
        y: Int[Tensor, " n"],
        weight: Float[Tensor, " v"] | None = None,
    ) -> Float[Tensor, ""]:
        """`own`, the head's loss over `logits` against `y`, blended with the raw-unit top-k term."""
        if not self.share:
            return own
        return (1 - self.share) * own + self.share * self.k * self.loss(logits, y, weight)

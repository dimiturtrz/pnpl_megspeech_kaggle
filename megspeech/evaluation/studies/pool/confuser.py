"""Who a row's nearest neighbours are: its own span's adjacent words, or genuine repeats from other spans.

Words ~0.25 s apart in 1 s windows share most of their signal, so adjacent words may be trivially the nearest
neighbours without being the same word. The diagnostic splits a pool graph's edges by that, and scores each part's
label purity.
"""

from dataclasses import dataclass, fields

import torch
from jaxtyping import Bool, Int
from torch import Tensor

NEAR = (1, 2)
"""Word offsets within a span that count as adjacent."""


@dataclass(frozen=True)
class Confusers:
    """Shares of a graph's edges by where the neighbour sits, and the label purity of each part."""

    same_span: float
    within_1: float
    within_2: float
    chance_span: float
    purity_all: float
    purity_same_span: float
    purity_within_2: float
    purity_outside: float

    @classmethod
    def of(
        cls, keep: Bool[Tensor, "d p p"], span: Int[Tensor, "d p"], place: Int[Tensor, "d p"], y: Int[Tensor, "d p"]
    ) -> "Confusers":
        """`keep`'s edges split by span: shares of all edges, and purity of the same-label edges in each part."""
        same = span[:, :, None] == span[:, None, :]
        near = same & ((place[:, :, None] - place[:, None, :]).abs() <= NEAR[-1])
        close = same & ((place[:, :, None] - place[:, None, :]).abs() <= NEAR[0])
        label = y[:, :, None] == y[:, None, :]
        rows = span.shape[1]
        others = same.sum() - same.shape[0] * rows

        def share(part: Bool[Tensor, "d p p"]) -> float:
            return float((keep & part).sum() / keep.sum().clamp(min=1))

        def purity(part: Bool[Tensor, "d p p"]) -> float:
            return float((keep & part & label).sum() / (keep & part).sum().clamp(min=1))

        everything = torch.ones_like(keep)
        return cls(
            share(same),
            share(close),
            share(near),
            float(others / (same.shape[0] * rows * (rows - 1))),
            purity(everything),
            purity(same),
            purity(near),
            purity(~same),
        )

    @staticmethod
    def mean(parts: list["Confusers"]) -> "Confusers":
        return Confusers(**{f.name: sum(getattr(p, f.name) for p in parts) / len(parts) for f in fields(Confusers)})

"""Token-level site dropout, training only."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


class SiteDrop(nn.Module):
    """Training-only token dropout: each window loses a Bernoulli share `p` of its sites, never all of them.
    Returns the keep mask [b, s], or None when inactive (eval or p = 0)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    @override
    def forward(self, h: Float[Tensor, "b s d t"]) -> Bool[Tensor, "b s"] | None:
        if not (self.training and self.p):
            return None
        scores = torch.rand(h.shape[:2], device=h.device)
        keep = scores >= self.p
        keep[torch.arange(len(keep), device=h.device), scores.argmax(1)] = True
        return keep

"""The identity across the site axis of a channel-first front end."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn


class NoAttention(nn.Module):
    """The identity across sites, taking the keep mask the attending layers would."""

    @override
    def forward(self, h: Float[Tensor, "b s d t"], keep: Bool[Tensor, "b s"] | None = None) -> Float[Tensor, "b s d t"]:
        return h

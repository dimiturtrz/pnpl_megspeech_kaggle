"""A gradient reversal with a runtime-set strength (megspeech-7gf.98)."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Reversal(nn.Module):
    """Identity forward; the gradient flowing back is scaled by `-strength`, a buffer a `Ramp` raises, so a compiled
    graph reads it as data."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("strength", torch.zeros(()))
        self.strength: Tensor

    @override
    def forward(self, x: Float[Tensor, "g w e"]) -> Float[Tensor, "g w e"]:
        held = x.detach()
        return held - self.strength * (x - held)

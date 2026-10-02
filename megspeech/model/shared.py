"""Every subject read through the same sensors: no per-subject parameters, the single-subject model."""

from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn


class Shared(nn.Module):
    """The sensors passed through unchanged, whichever subject heard the words."""

    @override
    def forward(self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"]) -> Float[Tensor, "r c t"]:
        return x

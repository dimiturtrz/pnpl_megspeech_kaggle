"""A strided convolution taking the trunk to half the time at a wider width: one step of a temporal pyramid."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.words.layers.norm import MaskedBatchNorm1d

KERNEL = 3
STRIDE = 2


class Downsample(nn.Module):
    def __init__(self, width_in: int, width_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(width_in, width_out, KERNEL, stride=STRIDE, padding=KERNEL // 2)
        self.norm = MaskedBatchNorm1d(width_out)
        self.act = nn.GELU()

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b g s"]:
        """`ceil(t / 2)` steps; `valid` rows alone set the batch statistics."""
        return self.act(self.norm(self.conv(x), valid))

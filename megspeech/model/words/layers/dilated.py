"""A residual dilated temporal convolution over hidden MEG features."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.words.layers.norm import MaskedBatchNorm1d

KERNEL = 3


class DilatedBlock(nn.Module):
    """A residual dilated convolution of kernel 3; at 50 Hz a dilation of 8 spans ~0.7 s."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        conv = nn.Conv1d(width, width, KERNEL, padding=dilation * (KERNEL - 1) // 2, dilation=dilation)
        self.body = nn.Sequential(conv, MaskedBatchNorm1d(width), nn.GELU(), nn.Dropout(dropout))

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """`valid` rows alone set the batch statistics; the rest are padding."""
        conv, norm, act, drop = self.body
        return x + drop(act(norm(conv(x), valid)))

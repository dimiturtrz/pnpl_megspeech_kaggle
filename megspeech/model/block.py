"""A residual dilated temporal convolution over hidden MEG features."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.norm import MaskedBatchNorm1d


class Block(nn.Module):
    """A residual dilated convolution; with 50 Hz input a dilation of 8 spans ~0.7 s."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            MaskedBatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """`valid` rows alone set the batch statistics; the rest are padding."""
        conv, norm, act, drop = self.body
        return x + drop(act(norm(conv(x), valid)))

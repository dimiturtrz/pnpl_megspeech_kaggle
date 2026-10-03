"""A residual dilated temporal convolution over hidden MEG features."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.words.layers.norm import MaskedBatchNorm1d


class DilatedBlock(nn.Module):
    """A residual dilated convolution, one per odd kernel size summed; at 50 Hz a dilation of 8 spans ~0.7 s.

    `gated` doubles the convolution's channels and halves them back with a GLU, one half gating the other: the
    activation of d'Ascoli et al.'s word-decoding trunk, in place of the GELU."""

    def __init__(
        self, width: int, dilation: int, dropout: float, kernels: tuple[int, ...] = (3,), *, gated: bool = False
    ) -> None:
        super().__init__()
        expand = 2 if gated else 1
        first, *rest = (self.conv(width, dilation, kernel, expand) for kernel in kernels)
        act = nn.GLU(dim=1) if gated else nn.GELU()
        self.body = nn.Sequential(first, MaskedBatchNorm1d(expand * width), act, nn.Dropout(dropout))
        self.parallel = nn.ModuleList(rest)

    @staticmethod
    def conv(width: int, dilation: int, kernel: int, expand: int = 1) -> nn.Conv1d:
        padding = dilation * (kernel - 1) // 2
        return nn.Conv1d(width, expand * width, kernel, padding=padding, dilation=dilation)

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """`valid` rows alone set the batch statistics; the rest are padding."""
        conv, norm, act, drop = self.body
        mixed = sum((branch(x) for branch in self.parallel), conv(x))
        return x + drop(act(norm(mixed, valid)))

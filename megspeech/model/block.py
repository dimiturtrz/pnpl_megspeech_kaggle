"""A residual dilated temporal convolution over hidden MEG features."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.norm import MaskedBatchNorm1d


class Block(nn.Module):
    """A residual dilated convolution, one per odd kernel size summed; at 50 Hz a dilation of 8 spans ~0.7 s."""

    def __init__(self, width: int, dilation: int, dropout: float, kernels: tuple[int, ...] = (3,)) -> None:
        super().__init__()
        first, *rest = (self.conv(width, dilation, kernel) for kernel in kernels)
        self.body = nn.Sequential(first, MaskedBatchNorm1d(width), nn.GELU(), nn.Dropout(dropout))
        self.parallel = nn.ModuleList(rest)

    @staticmethod
    def conv(width: int, dilation: int, kernel: int) -> nn.Conv1d:
        return nn.Conv1d(width, width, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation)

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """`valid` rows alone set the batch statistics; the rest are padding."""
        conv, norm, act, drop = self.body
        mixed = sum((branch(x) for branch in self.parallel), conv(x))
        return x + drop(act(norm(mixed, valid)))

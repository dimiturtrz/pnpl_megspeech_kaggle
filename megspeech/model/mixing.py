"""Gulati et al. 2020's convolution module over a span of words: neighbours mixed by a depthwise kernel."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn


class Mixing(nn.Module):
    """Pre-norm residual GLU, depthwise convolution over words and pointwise projection; padded words neither feed
    the convolution nor change."""

    def __init__(self, width: int, dropout: float, kernel: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.pointwise = nn.Linear(width, 2 * width)
        self.depthwise = nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width), nn.Dropout(dropout))

    @override
    def forward(self, h: Float[Tensor, "g w d"], padding: Bool[Tensor, "g w"]) -> Float[Tensor, "g w d"]:
        gated = nn.functional.glu(self.pointwise(self.norm(h)), dim=-1).masked_fill(padding[..., None], 0.0)
        mixed = self.out(self.depthwise(gated.transpose(1, 2)).transpose(1, 2).contiguous())
        return h + mixed.masked_fill(padding[..., None], 0.0)

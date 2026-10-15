"""Mean and max over equal time chunks: the default word readout."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class MeanMax(nn.Module):
    """Mean and max over each of `segments` equal time chunks."""

    def __init__(self, segments: int) -> None:
        super().__init__()
        self.segments = segments

    @override
    def forward(self, h: Float[Tensor, "b d t"]) -> Float[Tensor, "b e"]:
        pooled = [
            nn.functional.adaptive_avg_pool1d(h, self.segments),
            nn.functional.adaptive_max_pool1d(h, self.segments),
        ]
        return torch.cat([p.flatten(1) for p in pooled], dim=-1)

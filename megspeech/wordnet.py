"""A word classifier over one decimated MEG window: spatial mixing, dilated temporal convolutions, pooling."""

from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.block import Block

CHANNELS = 306
WORDS = 50


@dataclass(frozen=True)
class ConvConfig:
    hidden: int = 256
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2)
    dropout: float = 0.3


CONV = ConvConfig()


class WordNet(nn.Module):
    def __init__(self, config: ConvConfig = CONV) -> None:
        super().__init__()
        self.width = 2 * config.hidden
        self.spatial = nn.Conv1d(CHANNELS, config.hidden, 1)
        self.blocks = nn.Sequential(*(Block(config.hidden, d, config.dropout) for d in config.dilations))
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.width, WORDS))

    def embed(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b e"]:
        h = self.blocks(self.spatial(x))
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

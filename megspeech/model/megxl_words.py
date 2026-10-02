"""MEG-XL's trunk as the sentence model's word encoder, in place of `WordNet`."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import WORDS
from megspeech.model.lora import Lora
from megspeech.model.megxl import LATENT, Trunk
from megspeech.model.wordnet import ConvConfig, WordNet

POOLED = 2 * LATENT
"""Features per window: the trunk's mean and its max over every channel's tokens."""
COMPONENTS = 16
"""Whitened principal components of each channel's time-averaged token kept per window: 306 x 16 features, ~7.5 GB
bf16 over the training set's 765k windows, about the MEG cache's size."""
RESOLVED = CHANNELS * COMPONENTS
"""Features per window when every channel keeps its own."""


class Words(nn.Module):
    """A word classifier over one window from MEG-XL's features, pooled mean and max over its sensors and tokens;
    `WordNet`'s interface, so the sentence model reads either."""

    def __init__(self, trunk: Trunk, config: ConvConfig) -> None:
        super().__init__()
        Lora.inject(trunk, config.lora)
        self.trunk = trunk
        self.instance = config.instance
        self.width = POOLED
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.width, WORDS))

    @classmethod
    def pretrained(cls, root: DataRoot, config: ConvConfig) -> "Words":
        return cls(Trunk.pretrained(root), config)

    def embed(self, x: Float[Tensor, "b c t"], _valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        tokens = self.trunk(x).flatten(1, 2)
        return torch.cat([tokens.mean(1), tokens.amax(1)], dim=-1)

    def normalise(self, x: Float[Tensor, "b c t"], present: Bool[Tensor, "b 1 t"]) -> Float[Tensor, "b c t"]:
        return WordNet.standardise(x, present) if self.instance else x

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

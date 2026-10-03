"""MEG-XL's features as the sentence model's words: computed by its frozen trunk ahead of training
(`megspeech.training.extract`), then read by a head alone."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.constants import CHANNELS
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.trunk import LATENT, Trunk

POOLED = 2 * LATENT
"""Features per window: the trunk's mean and its max over every channel's tokens."""
COMPONENTS = 16
"""Whitened principal components of each channel's time-averaged token kept per window: 306 x 16 features, ~7.5 GB
bf16 over the training set's 765k windows, about the MEG cache's size."""
RESOLVED = CHANNELS * COMPONENTS
"""Features per window when every channel keeps its own."""


class FeatureWords(WordEncoder):
    """`MegxlWords` with its trunk run ahead of training: each window is its `width` saved features, read flat. The
    sentence model's scaler already standardised them per channel, so `instance` never applies twice."""

    def __init__(self, config: ConvConfig, width: int, *, instance: bool = False) -> None:
        super().__init__(width, instance=instance)
        self.head = self.classifier(config.dropout)

    @override
    def embed(self, x: Float[Tensor, "b c k"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        return x.flatten(1)


class MegxlWords(FeatureWords):
    """MEG-XL's features of one window, pooled mean and max over its sensors and tokens: what `extract` saves."""

    def __init__(self, trunk: Trunk, config: ConvConfig) -> None:
        super().__init__(config, POOLED, instance=config.instance)
        self.trunk = trunk

    @classmethod
    def pretrained(cls, root: DataRoot, config: ConvConfig) -> "MegxlWords":
        return cls(Trunk.pretrained(root), config)

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        tokens = self.trunk(x).flatten(1, 2)
        return torch.cat([tokens.mean(1), tokens.amax(1)], dim=-1)

"""Word encoders trained apart, read as one: each expert's CNN reads its own stretch of a shared window, as its own run
read it, so the experts fine-tune together under one context model rather than feeding it frozen embeddings
(megspeech-7gf.90)."""

from collections.abc import Sequence
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.input.scaler import Scaler
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.expert import Crop, Expert


class ExpertWords(WordEncoder):
    """The experts' embeddings side by side, standardised by `features` as the frozen embeddings were for the context
    model that warm-starts this one."""

    def __init__(self, config: ConvConfig, channels: int, crops: Sequence[Crop]) -> None:
        super().__init__(len(crops) * config.width, instance=False)
        self.experts = nn.ModuleList(Expert(crop, config, channels) for crop in crops)
        self.features = Scaler(torch.zeros(1, self.width, 1))
        self.head = self.classifier(config.dropout, config.head)

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        joined = torch.cat([expert.embed(x, valid) for expert in self.branches()], dim=-1)
        return self.features(joined[..., None])[..., 0]

    def branches(self) -> list[Expert]:
        return [expert for expert in self.experts if isinstance(expert, Expert)]

"""The word encoder over features a frozen MEG-XL trunk already pooled (`megspeech.data.features`)."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.vocabulary import WORDS
from megspeech.model.megxl_words import POOLED
from megspeech.model.wordnet import ConvConfig


class Cached(nn.Module):
    """`megxl_words.Words` with its frozen trunk run ahead of training: each window is its pooled features, one
    sample of `POOLED` channels, which the sentence model's scaler standardises per feature."""

    def __init__(self, config: ConvConfig) -> None:
        super().__init__()
        self.width = POOLED
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.width, WORDS))

    def embed(self, x: Float[Tensor, "b e 1"], _valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        return x[..., 0]

    def normalise(self, x: Float[Tensor, "b e 1"], _present: Bool[Tensor, "b 1 1"]) -> Float[Tensor, "b e 1"]:
        return x

    @override
    def forward(self, x: Float[Tensor, "b e 1"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

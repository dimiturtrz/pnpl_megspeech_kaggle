"""The word encoder over features a frozen MEG-XL trunk already computed (`megspeech.data.batching.features`)."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.wordnet import ConvConfig


class Cached(nn.Module):
    """`megxl_words.Words` with its frozen trunk run ahead of training: each window is its `width` features, which
    the sentence model's scaler has standardised per channel, read flat."""

    def __init__(self, config: ConvConfig, width: int) -> None:
        super().__init__()
        self.width = width
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.width, VOCAB_SIZE))

    def embed(self, x: Float[Tensor, "b c k"], _valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        return x.flatten(1)

    def normalise(self, x: Float[Tensor, "b c k"], _present: Bool[Tensor, "b 1 k"]) -> Float[Tensor, "b c k"]:
        return x

    @override
    def forward(self, x: Float[Tensor, "b c k"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

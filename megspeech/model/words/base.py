"""What the sentence model reads each word through: one window in, an embedding `width` wide and a word head out."""

from abc import ABC, abstractmethod
from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.words.heads import WordHead

EPSILON = 1e-6


class WordEncoder(nn.Module, ABC):
    """A word classifier over one window; subclasses build `head` from `classifier` after their own layers, so a
    seeded initialisation draws in the order checkpoints were trained with."""

    head: nn.Module

    def __init__(self, width: int, *, instance: bool) -> None:
        super().__init__()
        self.width = width
        self.instance = instance

    def classifier(self, dropout: float, head: WordHead = WordHead.LINEAR) -> nn.Module:
        return head.build(self.width, dropout)

    @abstractmethod
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        """`valid` rows alone set any batch statistics; the rest are padding."""

    def normalise(self, x: Float[Tensor, "b c t"], present: Bool[Tensor, "b 1 t"]) -> Float[Tensor, "b c t"]:
        """`x` standardised when `instance`, else unchanged."""
        return self.standardise(x, present) if self.instance else x

    @staticmethod
    def standardise(x: Float[Tensor, "b c t"], present: Bool[Tensor, "b 1 t"]) -> Float[Tensor, "b c t"]:
        """`x` z-scored per window and sensor over its `present` samples, zero elsewhere."""
        count = present.sum(-1, keepdim=True).clamp(min=1)
        centred = (x - (x * present).sum(-1, keepdim=True) / count) * present
        spread = (centred.square().sum(-1, keepdim=True) / count).sqrt()
        return centred / (spread + EPSILON)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

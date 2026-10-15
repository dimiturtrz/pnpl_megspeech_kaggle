"""What a word encoder's embedding is classified into the 50 words by: one linear layer, or one per part of speech."""

from collections.abc import Callable
from enum import StrEnum

from torch import nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.readout import WordReadout
from megspeech.model.words.tagged import ContextTaggedHead, TaggedHead


class LinearHead(nn.Sequential, WordReadout):
    """One linear layer over the 50 words."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__(nn.Dropout(dropout), nn.Linear(width, VOCAB_SIZE))


class WordHead(StrEnum):
    """The readout strategy of an encoder's word head."""

    LINEAR = "linear"
    TAGGED = "tagged"
    """`TaggedHead`: needs the tag auxiliary (`optim.tagging`) to learn its tags."""
    CONTEXT_TAGGED = "context-tagged"
    """`ContextTaggedHead`: its tags read the span-refined position; train with `optim.tap context`."""

    def build(self, width: int, dropout: float) -> WordReadout:
        return BUILDERS[self](width, dropout)


BUILDERS: dict[WordHead, Callable[[int, float], WordReadout]] = {
    WordHead.LINEAR: LinearHead,
    WordHead.TAGGED: TaggedHead,
    WordHead.CONTEXT_TAGGED: ContextTaggedHead,
}

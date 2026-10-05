"""What a word encoder's embedding is classified into the 50 words by: one linear layer, or one per part of speech."""

from collections.abc import Callable
from enum import StrEnum

from torch import nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.tagged import TaggedHead


class LinearHead(nn.Sequential):
    """One linear layer over the 50 words."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__(nn.Dropout(dropout), nn.Linear(width, VOCAB_SIZE))


class WordHead(StrEnum):
    """The readout strategy of an encoder's word head."""

    LINEAR = "linear"
    TAGGED = "tagged"
    """`TaggedHead`: needs the tag auxiliary (`optim.tagging`) to learn its tags."""

    def build(self, width: int, dropout: float) -> nn.Module:
        return BUILDERS[self](width, dropout)


BUILDERS: dict[WordHead, Callable[[int, float], nn.Module]] = {WordHead.LINEAR: LinearHead, WordHead.TAGGED: TaggedHead}

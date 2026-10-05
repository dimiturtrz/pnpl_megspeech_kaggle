"""What a word encoder's embedding is classified into the 50 words by: one linear layer, or one per part of speech."""

from collections.abc import Callable
from enum import StrEnum
from typing import cast

from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.tagged import ContextTaggedHead, TaggedHead


class LinearHead(nn.Sequential):
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

    def build(self, width: int, dropout: float) -> nn.Module:
        return BUILDERS[self](width, dropout)

    def attach(self, head: nn.Module, context_width: int) -> None:
        """Size `head` for the span-refined positions of `context_width` it may read."""
        ATTACHERS[self](head, context_width)

    def read(
        self, head: nn.Module, slots: Float[Tensor, "g w e"], refined: Float[Tensor, "g w h"]
    ) -> Float[Tensor, "g w v"]:
        """`head`'s words over the word embeddings `slots`, given the `refined` positions it may read."""
        return READERS[self](head, slots, refined)


BUILDERS: dict[WordHead, Callable[[int, float], nn.Module]] = {
    WordHead.LINEAR: LinearHead,
    WordHead.TAGGED: TaggedHead,
    WordHead.CONTEXT_TAGGED: ContextTaggedHead,
}
ATTACHERS: dict[WordHead, Callable[[nn.Module, int], None]] = {
    WordHead.LINEAR: lambda _, __: None,
    WordHead.TAGGED: lambda _, __: None,
    WordHead.CONTEXT_TAGGED: lambda head, width: cast("ContextTaggedHead", head).attach(width),
}
READERS: dict[WordHead, Callable[[nn.Module, Tensor, Tensor], Tensor]] = {
    WordHead.LINEAR: lambda head, slots, _: head(slots),
    WordHead.TAGGED: lambda head, slots, _: head(slots),
    WordHead.CONTEXT_TAGGED: lambda head, slots, refined: head(slots, refined),
}

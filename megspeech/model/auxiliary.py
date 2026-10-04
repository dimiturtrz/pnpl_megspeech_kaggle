"""The sentence model trained beside auxiliary heads on its word embeddings or its span-refined positions."""

from enum import StrEnum
from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.sentence import SentenceNet


class Tap(StrEnum):
    """Which per-word representation the auxiliary heads read."""

    WORD = "word"
    """The encoder's word embedding, before the span transformer."""
    CONTEXT = "context"
    """The span transformer's output at the word, what the context head reads (d'Ascoli et al.'s loss placement)."""

    def width(self, net: SentenceNet) -> int:
        """Width of the representation this tap reads off `net`."""
        return {Tap.WORD: net.words.width, Tap.CONTEXT: net.width}[self]

    def pick(self, slots: Float[Tensor, "g w e"], refined: Float[Tensor, "g w h"]) -> Float[Tensor, "g w _"]:
        """This tap's representation of every span position."""
        return {Tap.WORD: slots, Tap.CONTEXT: refined}[self]


class WithAuxiliary(nn.Module):
    """The sentence model with auxiliary heads reading its `tap`, trained as one."""

    def __init__(self, net: SentenceNet, heads: nn.ModuleList, tap: Tap = Tap.WORD) -> None:
        super().__init__()
        self.net = net
        self.heads = heads
        self.tap = tap

    @override
    def forward(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w _"], ...]:
        """`SentenceNet`'s two heads, then each auxiliary head's reading of every span position's tapped
        representation."""
        refined, slots, context, word = self.net.heard(x, onset, slot, subject)
        tapped = self.tap.pick(slots, refined)
        return context, word, *(head(tapped) for head in self.heads)

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: `SentenceNet`'s, then the auxiliary heads."""
        return *self.net.readouts(), *self.heads

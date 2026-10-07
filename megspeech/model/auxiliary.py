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
    TIME = "time"
    """The encoder's per-timestep features before pooling, `[h t]` a word, for the heads that read the window's time
    course."""

    @staticmethod
    def pooled() -> tuple["Tap", ...]:
        """The taps that hold one vector per word."""
        return Tap.WORD, Tap.CONTEXT

    def width(self, net: SentenceNet) -> int:
        """Width of the representation this tap reads off `net`."""
        return {Tap.WORD: net.words.width, Tap.CONTEXT: net.width, Tap.TIME: net.words.time_width()}[self]

    def pick(
        self,
        slots: Float[Tensor, "g w e"],
        refined: Float[Tensor, "g w h"],
        times: Float[Tensor, "g w k t"] | None = None,
    ) -> Float[Tensor, "g w _"] | Float[Tensor, "g w k t"]:
        """This tap's representation of every span position; `times` is read only by `TIME`."""
        if self is Tap.TIME:
            if times is None:
                raise ValueError("the time tap reads the window's time course: pass `times`")
            return times
        return {Tap.WORD: slots, Tap.CONTEXT: refined}[self]


class WithAuxiliary(nn.Module):
    """The sentence model with auxiliary heads reading its `tap`, trained as one."""

    def __init__(
        self, net: SentenceNet, heads: nn.ModuleList, tap: Tap = Tap.WORD, taps: tuple[Tap, ...] | None = None
    ) -> None:
        """`taps` names each head's own tap; heads without one read `tap`."""
        super().__init__()
        self.net = net
        self.heads = heads
        self.tap = tap
        self.taps = taps or tuple(tap for _ in heads)

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
        if Tap.TIME in self.taps:
            refined, slots, context, word, times = self.net.heard_timed(x, onset, slot, subject)
        else:
            (refined, slots, context, word), times = self.net.heard(x, onset, slot, subject), None
        return (
            context,
            word,
            *(head(tap.pick(slots, refined, times)) for head, tap in zip(self.heads, self.taps, strict=True)),
        )

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: `SentenceNet`'s, then the auxiliary heads."""
        return *self.net.readouts(), *self.heads

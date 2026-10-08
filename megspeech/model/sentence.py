"""The sentence-context model: per-word embeddings, onset timing, and a transformer across the span."""

from typing import override

from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.constants import CHANNELS
from megspeech.model.config import CONTEXT, ContextConfig
from megspeech.model.context.timing import TimingHead
from megspeech.model.input.subjects import Unseen
from megspeech.model.mixer import Mixer
from megspeech.model.readout import ContextReadout
from megspeech.model.span import Span


class SentenceNet(Span):
    """Per-word embeddings refined by a transformer over the span of words heard together.

    Returns the context head's logits and the word head's, both per word slot, so one model is both the
    isolated-word classifier and the sentence-context one.
    """

    def __init__(
        self,
        scale: nn.Module,
        context: ContextConfig = CONTEXT,
        prototypes: Float[Tensor, "v d"] | None = None,
        unseen: Unseen = Unseen.IDENTITY,
        *,
        channels: int = CHANNELS,
    ) -> None:
        """`prototypes` are the fixed per-word anchors the context head reads against; see `Span` for the rest."""
        super().__init__(scale, context, unseen, channels=channels)
        self.reading = context.conv.head
        self.reading.attach(self.words.head, context.width)
        self.context = Mixer.of(context)
        self.head = ContextReadout.build(context, prototypes)
        self.timing = TimingHead.over_vocabulary(active=context.rhythm)

    @override
    def forward(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """`slot` is each span position's row of `x`; `len(x)` marks an empty position, and rows no slot names are
        padding that neither normalises nor reaches the transformer."""
        _, _, context, word = self.heard(x, onset, slot, subject)
        return context, word

    def heard(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"], Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The refined positions and the word embeddings, then the context head's logits with the timing head's
        added, and the word head's."""
        h, slots = self.features(x, onset, slot, subject)
        return self.finish(h, slots, onset, slot < len(x))

    def heard_timed(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[
        Float[Tensor, "g w h"],
        Float[Tensor, "g w e"],
        Float[Tensor, "g w v"],
        Float[Tensor, "g w v"],
        Float[Tensor, "g w k t"],
    ]:
        """`heard`'s outputs and, last, the encoder's per-timestep features at each position: `heard` itself, the
        encoder read once."""
        slots, times = self.words.positions_timed(self.prepare(x, subject), slot)
        h = self.chunked(x, onset, slot, subject, slots)
        return *self.finish(h, slots, onset, slot < len(x)), times

    def finish(
        self,
        h: Float[Tensor, "g w h"],
        slots: Float[Tensor, "g w e"],
        onset: Float[Tensor, "g w"],
        mask: Bool[Tensor, "g w"],
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"], Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """`heard`'s outputs from the transformer's input `h` and the word embeddings `slots`."""
        refined = self.refine(h, mask)
        context, word = self.read(refined, slots)
        return refined, slots, context + self.timing(onset, mask), word

    def word_logits(self, embeddings: Float[Tensor, "r e"]) -> Float[Tensor, "r v"]:
        """The stage-1 word head on `embeddings`."""
        return self.words.head(embeddings)

    def classify(
        self, h: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over `h` refined across its span's `mask`ed positions, and the word head's."""
        return self.read(self.refine(h, mask), slots)

    def refine(self, h: Float[Tensor, "g w h"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w h"]:
        """`h`, each word's `tokens` in a row, refined across its span's `mask`ed words and averaged back to one per
        word: what the context head reads."""
        tokens = mask.repeat_interleave(self.tokens, dim=1)
        return self.context(h, src_key_padding_mask=~tokens).unflatten(1, (-1, self.tokens)).mean(2)

    def read(
        self, refined: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over the `refined` positions, and the word head's over their `slots`."""
        return self.head(refined), self.reading.read(self.words.head, slots, refined)

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: the context head, then the word head."""
        return self.head, self.words.head

"""The sentence-context model: per-word embeddings, onset timing, and a transformer across the span."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput
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
    def forward(self, spans: SpanInput) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """Rows of `spans.x` no slot names are padding that neither normalises nor reaches the transformer."""
        _, _, context, word = self.heard(spans)
        return context, word

    def heard(
        self, spans: SpanInput
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"], Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The refined positions and the word embeddings, then the context head's logits with the timing head's
        added, and the word head's."""
        h, slots = self.features(spans)
        return self.finish(h, slots, spans.onset, spans.mask)

    def heard_timed(
        self, spans: SpanInput
    ) -> tuple[
        Float[Tensor, "g w h"],
        Float[Tensor, "g w e"],
        Float[Tensor, "g w v"],
        Float[Tensor, "g w v"],
        Float[Tensor, "g w k t"],
    ]:
        """`heard`'s outputs and, last, the encoder's per-timestep features at each position: `heard` itself, the
        encoder read once."""
        slots, times = self.words.positions_timed(self.prepare(spans.x, spans.subject), spans.slot)
        h = self.chunked(spans, slots)
        return *self.finish(h, slots, spans.onset, spans.mask), times

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
        """`h` refined across its span's `mask`ed words: what the context head reads."""
        return self.context(h, src_key_padding_mask=~mask)

    def read(
        self, refined: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over the `refined` positions, and the word head's over their `slots`."""
        return self.head(refined), self.reading.read(self.words.head, slots, refined)

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: the context head, then the word head."""
        return self.head, self.words.head

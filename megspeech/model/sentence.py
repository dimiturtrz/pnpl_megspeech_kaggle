"""The sentence-context model: per-word embeddings, onset timing, and a transformer across the span."""

from typing import Self, override

from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.model.config import CONTEXT, ContextConfig
from megspeech.model.context.timing import TimingHead
from megspeech.model.head.readout import ContextReadout
from megspeech.model.input.subjects import Unseen
from megspeech.model.mixer import Mixer
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
        self.words.attach(context.width)
        self.context = Mixer.of(context)
        self.head = ContextReadout.build(context, prototypes)
        self.timing = TimingHead.over_vocabulary(active=context.rhythm)
        pyramid = context.conv.pyramid
        self.pyramid = pyramid.readout(self.words, context.width)
        self.hierarchy = ContextReadout.hierarchy(context)
        self.train()

    @override
    def train(self, mode: bool = True) -> Self:
        """`mode` set throughout, but for what the pyramid readout holds frozen."""
        super().train(mode)
        self.pyramid.hold(self)
        return self

    def warm(self, state: dict[str, Tensor]) -> None:
        """Another run's `state` loaded in whole; only the pyramid readout this model adds may be missing from it."""
        missing, unexpected = self.load_state_dict(state, strict=False)
        stale = set(missing) - self.pyramid.fresh()
        if stale or unexpected:
            raise KeyError(f"warm state lacks {sorted(stale)} and has unknown {sorted(unexpected)}")

    @override
    def forward(self, spans: SpanInput) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """Rows of `spans.x` no slot names are padding that neither normalises nor reaches the transformer."""
        _, _, context, word = self.heard(spans)
        return context, word

    def heard(
        self, spans: SpanInput
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"], Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The refined positions and the word embeddings, then the context head's logits with the timing head's
        added, and the word head's, both corrected by the pyramid readout."""
        slots, taps = self.pyramid.positions(self.words, self.prepared(spans))
        slots = self.centred(spans, slots)
        refined, slots, context, word = self.finish(self.chunked(spans, slots), slots, spans.onset, spans.mask)
        return refined, slots, *self.pyramid.correct(taps, refined, context, word)

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
        encoder read once, for a model with no pyramid readout (`RUN_RULES`)."""
        slots, times = self.words.positions_timed(self.prepare(spans.x, spans.subject), spans.slot)
        slots = self.centred(spans, slots)
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
        """`h` refined across its span's `mask`ed words: what the context head reads.

        Attention runs on the math kernel: under `torch.compile` the memory-efficient kernel's backward rejects the
        padding mask's layout ("(*bias): last dimension must be contiguous"), and spans are too short for it to matter.
        """
        with sdpa_kernel(SDPBackend.MATH):
            return self.context(h, src_key_padding_mask=~mask)

    def read(
        self, refined: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over the `refined` positions, and the word head's over their `slots`."""
        return self.hierarchy(self.head(refined), refined), self.words.classify(slots, refined)

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: the context head, then the word head."""
        return self.head, self.words.head

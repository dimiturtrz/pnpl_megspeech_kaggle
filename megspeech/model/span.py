"""The span input: per-word embeddings and onset timing assembled into the transformer's input."""

from typing import Self

from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.model.config import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.model.context.intervals import NextDrop
from megspeech.model.context.masking import WordMask
from megspeech.model.context.timing import Jitter
from megspeech.model.encoder.encoding import Encoding
from megspeech.model.encoder.fusion import Fusion
from megspeech.model.input.subjects import Unseen
from megspeech.model.words.layers.centring import RunCentring


class Span(Encoding):
    """The stage-1 encoding plus everything that turns a span of word embeddings into the transformer's input."""

    def __init__(
        self,
        scale: nn.Module,
        context: ContextConfig = CONTEXT,
        unseen: Unseen = Unseen.IDENTITY,
        *,
        channels: int = CHANNELS,
    ) -> None:
        """`unseen` reads a subject the subject layer has no matrix for (`SubjectLayer`); `channels` are the rows a
        recording reaches the model in: the sensors, or a `Projection`'s components."""
        super().__init__(scale, context, channels, unseen)
        self.project = nn.Linear(self.words.width, context.width)
        self.fusion = Fusion((), context.branch_drop)
        self.clock = context.clock
        self.onsets = OnsetEncoding(context)
        self.intervals = context.intervals
        self.jitter = Jitter(context.onset_jitter)
        self.next_drop = NextDrop(context.next_drop)
        self.masking = WordMask(context.width, context.word_mask)
        self.centring = RunCentring(self.words.width, active=context.run_centred)

    def fuse(self, fusion: Fusion) -> Self:
        """This model reading `fusion`'s encoders beside its own: their features join its at the transformer's
        input."""
        self.fusion = fusion
        return self

    def features(self, spans: SpanInput) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"]]:
        """The transformer's input at each span position, and the position's word embedding (zeros when empty)."""
        slots = self.centred(spans, self.words.positions(self.prepare(spans.x, spans.subject), spans.onset, spans.slot))
        return self.chunked(spans, slots), slots

    def centred(self, spans: SpanInput, slots: Float[Tensor, "g w e"]) -> Float[Tensor, "g w e"]:
        """The word embeddings `slots` of `spans` as the heads read them: centred on their run when `run_centred`."""
        return self.centring(slots, spans.mask, spans.runs)

    def chunked(self, spans: SpanInput, slots: Float[Tensor, "g w e"]) -> Float[Tensor, "g w h"]:
        """The transformer's input at each span position from the word embeddings `slots`."""
        mask = spans.mask
        drop = self.next_drop if self.training else NextDrop()
        if self.training:
            spans = spans.timed(self.jitter.apply(spans.onset, mask))
        intervals = drop.apply(self.intervals, spans.onset, mask, spans.finish)
        timing = self.onsets(self.clock.times(spans.onset, mask), intervals)
        own = self.project(slots)
        others = (b.projected(spans) for b in self.fusion.branches)
        return self.masking(self.fusion.mix([own, *others]), mask) + timing

    @property
    def width(self) -> int:
        """Width of the refined positions the context head reads."""
        return self.project.out_features

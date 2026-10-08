"""The span input: per-word embeddings, onset timing and latency tags assembled into the transformer's input."""

from typing import Self

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.constants import CHANNELS
from megspeech.model.config import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.model.context.masking import WordMask
from megspeech.model.context.timing import Jitter
from megspeech.model.encoding import Encoding
from megspeech.model.fusion import Fusion
from megspeech.model.input.subjects import Unseen
from megspeech.model.words.layers.mean_max import MeanMax

LATENCY_INIT = 0.02
"""Spread of each latency tag at initialisation, a transformer's usual position-embedding scale (BERT, GPT-2)."""


class Span(Encoding):
    """The stage-1 encoding plus everything that turns a span of word embeddings into the transformer's input."""

    latency: Tensor

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
        self.tokens = context.tokens
        self.project = nn.Linear(self.words.width // self.tokens, context.width)
        self.fusion = Fusion((), context.branch_drop)
        self.clock = context.clock
        self.onsets = OnsetEncoding(context)
        self.intervals = context.intervals
        self.jitter = Jitter(context.onset_jitter)
        self.masking = WordMask(context.width, context.word_mask)
        self._tag(context)

    def fuse(self, fusion: Fusion) -> Self:
        """This model reading `fusion`'s encoders beside its own: their features join its at the transformer's
        input."""
        self.fusion = fusion
        return self

    def _tag(self, context: ContextConfig) -> None:
        """Each latency token's learned tag as `latency`; with one token per word a zero outside the state, so the
        checkpoints trained before it load, and their initialisation draws no more."""
        if context.latency:
            self.latency = nn.Parameter(torch.randn(self.tokens, context.width) * LATENCY_INIT)
        else:
            self.register_buffer("latency", torch.zeros(1, context.width), persistent=False)

    def features(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"]]:
        """The transformer's input at each span position, and the position's word embedding (zeros when empty)."""
        slots = self.words.positions(self.prepare(x, subject), onset, slot)
        return self.chunked(x, onset, slot, subject, slots), slots

    def chunked(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
        slots: Float[Tensor, "g w e"],
    ) -> Float[Tensor, "g w h"]:
        """The transformer's input at each span position from the word embeddings `slots`."""
        mask = slot < len(x)
        onset = self.jitter.apply(onset, mask) if self.training else onset
        timing = self.onsets(self.clock.times(onset, mask), self.intervals.features(onset, mask))
        own = self.project(MeanMax.segmented(slots, self.tokens))
        others = (b.projected(x, subject, onset, slot, self.tokens) for b in self.fusion.branches)
        content = self.masking(self.fusion.mix([own, *others]), mask)
        return (content + timing[..., None, :] + self.latency).flatten(1, 2)

    @property
    def width(self) -> int:
        """Width of the refined positions the context head reads."""
        return self.project.out_features

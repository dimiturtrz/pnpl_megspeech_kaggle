"""The sentence-context model: per-word embeddings, onset timing, and a transformer across the span."""

from pathlib import Path
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.constants import CHANNELS
from megspeech.model.config import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.model.context.span import SpanConformer
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import SubjectLayer, Unseen

SCALE_MEDIAN = "scale.median"
"""A SentenceNet state's per-channel scaler median: one entry per channel the model reads."""


class SentenceNet(nn.Module):
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
    ) -> None:
        """`unseen` reads a subject the subject layer has no matrix for (`SubjectLayer`)."""
        super().__init__()
        self.scale = scale
        self.subjects = SubjectLayer.of(context.subjects, CHANNELS, unseen)
        self.words = context.encoder.build(context.conv)
        self.project = nn.Linear(self.words.width, context.width)
        self.clock = context.clock
        self.onsets = OnsetEncoding(context)
        self.intervals = context.intervals
        self.context = self.mixer(context)
        self.head = context.objective.head(context.width, prototypes)

    @classmethod
    def load(cls, path: Path, unseen: Unseen = Unseen.IDENTITY) -> "SentenceNet":
        """The checkpoint at `path` in the shape its directory's config records; its state overwrites a placeholder
        scaler shaped as the state's own: MEG channels, or a cached encoder's feature channels."""
        state = torch.load(path, map_location="cpu")
        placeholder = Scaler(torch.zeros(1, state[SCALE_MEDIAN].shape[0], 1))
        model = cls(placeholder, context=ContextConfig.load(path.parent), unseen=unseen)
        model.load_state_dict(state)
        return model

    @staticmethod
    def mixer(context: ContextConfig) -> nn.Module:
        """The layers that refine word embeddings across the span: a transformer, conformer when `mixing`."""
        if context.mixing:
            return SpanConformer(context.width, context.heads, context.layers, context.dropout, context.mixing)
        layer = SpanConformer.layer(context.width, context.heads, context.dropout)
        norm = nn.LayerNorm(context.width) if context.final_norm else None
        return nn.TransformerEncoder(layer, context.layers, norm=norm, enable_nested_tensor=False)

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
        return self.classify(*self.features(x, onset, slot, subject), slot < len(x))

    def features(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"]]:
        """The transformer's input at each span position, and the position's word embedding (zeros when empty)."""
        rows = len(x)
        valid = torch.zeros(rows + 1, dtype=torch.bool, device=x.device).index_fill_(0, slot.flatten(), value=True)
        embedded = self.embed(x, subject, valid[:rows])
        slots = nn.functional.pad(embedded, (0, 0, 0, 1))[slot]
        mask = slot < rows
        timing = self.onsets(self.clock.times(onset, mask), self.intervals.features(onset, mask))
        return self.project(slots) + timing, slots

    def embed(
        self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"], valid: Bool[Tensor, " r"] | None = None
    ) -> Float[Tensor, "r e"]:
        """Each window's word embedding through the scaler, the subject layer and the encoder; `valid` rows alone
        set any batch statistics."""
        scaled = self.words.normalise(self.subjects(self.scale(x), subject), x.ne(0).any(dim=1, keepdim=True))
        return self.words.embed(scaled, valid)

    def classify(
        self, h: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over `h` refined across its span's `mask`ed positions, and the word head's."""
        return self.read(self.refine(h, mask), slots)

    def refine(self, h: Float[Tensor, "g w h"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w h"]:
        """`h` refined across its span's `mask`ed positions: what the context head reads."""
        return self.context(h, src_key_padding_mask=~mask)

    def read(
        self, refined: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over the `refined` positions, and the word head's over their `slots`."""
        return self.head(refined), self.words.head(slots)

    @property
    def width(self) -> int:
        """Width of the refined positions the context head reads."""
        return self.project.out_features

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: the context head, then the word head."""
        return self.head, self.words.head

"""The sentence-context model: per-word CNN embeddings, onset timing, and a transformer across the span."""

from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.runs import CHANNELS
from megspeech.model.onsets import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.model.span import SpanConformer
from megspeech.model.subjects import SubjectLayer, Unseen


class SentenceNet(nn.Module):
    """Per-word CNN embeddings refined by a transformer over the span of words heard together.

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

    @staticmethod
    def mixer(context: ContextConfig) -> nn.Module:
        """The layers that refine word embeddings across the span: a transformer, conformer when `mixing`."""
        if context.mixing:
            return SpanConformer(context.width, context.heads, context.layers, context.dropout, context.mixing)
        layer = nn.TransformerEncoderLayer(
            context.width, context.heads, 4 * context.width, context.dropout, batch_first=True, norm_first=True
        )
        return nn.TransformerEncoder(layer, context.layers, enable_nested_tensor=False)

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
        scaled = self.words.normalise(self.subjects(self.scale(x), subject), x.ne(0).any(dim=1, keepdim=True))
        embedded = self.words.embed(scaled, valid[:rows])
        slots = nn.functional.pad(embedded, (0, 0, 0, 1))[slot]
        mask = slot < rows
        timing = self.onsets(self.clock.times(onset, mask), self.intervals.features(onset, mask))
        return self.project(slots) + timing, slots

    def classify(
        self, h: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over `h` refined across its span's `mask`ed positions, and the word head's."""
        return self.head(self.context(h, src_key_padding_mask=~mask)), self.words.head(slots)

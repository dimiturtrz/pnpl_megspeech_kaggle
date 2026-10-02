"""The sentence-context model: per-word CNN embeddings, onset timing, and a transformer across the span."""

from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.onsets import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.model.wordnet import CONV, ConvConfig, WordNet


class SentenceNet(nn.Module):
    """Per-word CNN embeddings refined by a transformer over the span of words heard together.

    Returns the context head's logits and the word head's, both per word slot, so one model is both the
    isolated-word classifier and the sentence-context one.
    """

    def __init__(
        self,
        scale: nn.Module,
        conv: ConvConfig = CONV,
        context: ContextConfig = CONTEXT,
        prototypes: Float[Tensor, "v d"] | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.words = WordNet(conv)
        self.project = nn.Linear(self.words.width, context.width)
        self.clock = context.clock
        self.onsets = OnsetEncoding(context)
        layer = nn.TransformerEncoderLayer(
            context.width, context.heads, 4 * context.width, context.dropout, batch_first=True, norm_first=True
        )
        self.context = nn.TransformerEncoder(layer, context.layers, enable_nested_tensor=False)
        self.head = context.objective.head(context.width, prototypes)

    @override
    def forward(
        self, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """`slot` is each span position's row of `x`; `len(x)` marks an empty position, and rows no slot names are
        padding that neither normalises nor reaches the transformer."""
        rows = len(x)
        mask = slot < rows
        valid = torch.zeros(rows + 1, dtype=torch.bool, device=x.device).index_fill_(0, slot.flatten(), value=True)
        embedded = self.words.embed(self.scale(x), valid[:rows])
        slots = nn.functional.pad(embedded, (0, 0, 0, 1))[slot]
        h = self.project(slots) + self.onsets(self.clock.times(onset, mask))
        return self.head(self.context(h, src_key_padding_mask=~mask)), self.words.head(slots)

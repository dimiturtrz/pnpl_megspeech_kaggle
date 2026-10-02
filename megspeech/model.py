"""The sentence-context model: per-word CNN embeddings, onset timing, and a transformer across the span."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.onsets import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.wordnet import CONV, WORDS, ConvConfig, WordNet


class SentenceNet(nn.Module):
    """Per-word CNN embeddings refined by a transformer over the span of words heard together.

    Returns the context head's logits and the word head's, both per word slot, so one model is both the
    isolated-word classifier and the sentence-context one.
    """

    def __init__(self, scale: nn.Module, conv: ConvConfig = CONV, context: ContextConfig = CONTEXT) -> None:
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
        self.head = nn.Linear(context.width, WORDS)

    @override
    def forward(
        self, x: Float[Tensor, "n c t"], onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        embedded = self.words.embed(self.scale(x))
        slots = embedded.new_zeros(*mask.shape, embedded.shape[-1])
        slots[mask] = embedded
        h = self.project(slots) + self.onsets(self.clock.times(onset, mask))
        return self.head(self.context(h, src_key_padding_mask=~mask)), self.words.head(slots)

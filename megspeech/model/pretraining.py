"""The sentence model trained beside auxiliary heads on its word embeddings."""

from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.sentence import SentenceNet


class Pretraining(nn.Module):
    """The sentence model with auxiliary heads reading its word embeddings, trained as one."""

    def __init__(self, net: SentenceNet, heads: nn.ModuleList) -> None:
        super().__init__()
        self.net = net
        self.heads = heads

    @override
    def forward(
        self, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w _"], ...]:
        """`SentenceNet`'s two heads, then each auxiliary head's reading of every span position's word embedding."""
        h, slots = self.net.features(x, onset, slot)
        context, word = self.net.classify(h, slots, slot < len(x))
        return context, word, *(head(slots) for head in self.heads)

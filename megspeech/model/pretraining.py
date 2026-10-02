"""The sentence model trained beside a semantic projection of its word embeddings."""

from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.sentence import SentenceNet
from megspeech.objective.semantic import Semantic


class Pretraining(nn.Module):
    """The sentence model with a semantic projection of its word embeddings, trained as one."""

    def __init__(self, net: SentenceNet, semantic: Semantic) -> None:
        super().__init__()
        self.net = net
        self.semantic = semantic

    @override
    def forward(
        self, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"], Float[Tensor, "g w d"]]:
        """`SentenceNet`'s two heads, and each span position's word embedding projected into t5's space."""
        h, slots = self.net.features(x, onset, slot)
        context, word = self.net.classify(h, slots, slot < len(x))
        return context, word, self.semantic(slots)

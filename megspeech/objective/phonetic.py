"""The phonetic auxiliary: each heard word's embedding classified into the leading phonemes of its text."""

from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.phonemes import PHONEMES, UNKNOWN


class Phonetic(nn.Module):
    """A linear readout of word embeddings over the 39 phonemes at each of `positions` leading positions,
    position-major."""

    def __init__(self, width: int, positions: int = 1) -> None:
        super().__init__()
        self.classify = nn.Linear(width, positions * len(PHONEMES))

    @override
    def forward(self, embedded: Float[Tensor, "*b w"]) -> Float[Tensor, "*b q"]:
        return self.classify(embedded)

    @staticmethod
    def loss(logits: Float[Tensor, "n p"], phonemes: Int[Tensor, " n"]) -> Float[Tensor, ""]:
        """Cross-entropy over the rows with a known phoneme; 0 when there are none, without a host sync."""
        rows = functional.cross_entropy(logits.float(), phonemes.clamp(min=0), reduction="none") * (phonemes != UNKNOWN)
        return rows.sum() / (phonemes != UNKNOWN).sum().clamp(min=1)

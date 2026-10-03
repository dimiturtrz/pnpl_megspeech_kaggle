"""The phonetic auxiliary: each heard word's embedding classified into the leading phonemes of its text."""

from dataclasses import dataclass
from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.labels.phonemes import PHONEMES, Pronunciation
from megspeech.objective.term import Classified


class Phonetic(nn.Module):
    """A linear readout of word embeddings over the 39 phonemes at each of `positions` leading positions,
    position-major."""

    def __init__(self, width: int, positions: int = 1) -> None:
        super().__init__()
        self.classify = nn.Linear(width, positions * len(PHONEMES))

    @override
    def forward(self, embedded: Float[Tensor, "*b w"]) -> Float[Tensor, "*b q"]:
        return self.classify(embedded)


@dataclass(frozen=True)
class PhoneticTerm(Classified):
    """Every heard word's window, in or out of the 50, classified into its first `positions` phonemes
    (`Phonetic`)."""

    weight: float
    pronunciation: Pronunciation
    positions: int = 1

    @override
    def head(self, width: int) -> nn.Module:
        return Phonetic(width, self.positions).to(self.pronunciation.spelled.device)

    @override
    def count(self) -> int:
        return len(PHONEMES)

    @override
    def classify(self, token: Int[Tensor, "g w"]) -> Int[Tensor, "g w p"]:
        return self.pronunciation.phonemes(token)[..., : self.positions]

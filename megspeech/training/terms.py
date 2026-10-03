"""The auxiliary terms, each a head over every heard word's embedding and its weighted loss: the word's t5
embedding aligned, its first phoneme classified."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.labels.phonemes import PHONEMES, Pronunciation
from megspeech.objective.phonetic import Phonetic
from megspeech.objective.prototypes import Lexicon
from megspeech.objective.semantic import Semantic


class Term(ABC):
    """One auxiliary head over the word embeddings and the weighted loss of its output."""

    @abstractmethod
    def head(self, width: int) -> nn.Module: ...

    @abstractmethod
    def loss(self, head: nn.Module, output: Float[Tensor, "g w _"], tokens: Int[Tensor, "g w"]) -> Float[Tensor, ""]:
        """`output` of `head` at span positions whose words are `tokens` (`Split.token`)."""


@dataclass(frozen=True)
class Aligned(Term):
    """Every heard word's window, in or out of the 50, trained toward its t5-large embedding (`Semantic`)."""

    weight: float
    lexicon: Lexicon

    @override
    def head(self, width: int) -> nn.Module:
        return Semantic(width).to(self.lexicon.vectors.device)

    @override
    def loss(self, head: nn.Module, output: Float[Tensor, "g w _"], tokens: Int[Tensor, "g w"]) -> Float[Tensor, ""]:
        kinds = self.lexicon.rows(tokens).flatten()
        targets = self.lexicon.vectors[kinds.clamp(min=0)]
        return self.weight * cast("Semantic", head).loss(output.flatten(0, 1), targets, kinds)


@dataclass(frozen=True)
class Lexical(Term):
    """Every heard word's window, in or out of the 50, classified into its word among every distinct word heard
    (the `Lexicon`'s keys): the 60% of windows the heads never label get a word identity to discriminate."""

    weight: float
    lexicon: Lexicon

    @override
    def head(self, width: int) -> nn.Module:
        return nn.Linear(width, len(self.lexicon.keys)).to(self.lexicon.keys.device)

    @override
    def loss(self, head: nn.Module, output: Float[Tensor, "g w _"], tokens: Int[Tensor, "g w"]) -> Float[Tensor, ""]:
        return self.weight * Phonetic.loss(output.flatten(0, 1), self.lexicon.rows(tokens).flatten())


@dataclass(frozen=True)
class Pronounced(Term):
    """Every heard word's window, in or out of the 50, classified into its first `positions` phonemes
    (`Phonetic`)."""

    weight: float
    pronunciation: Pronunciation
    positions: int = 1

    @override
    def head(self, width: int) -> nn.Module:
        return Phonetic(width, self.positions).to(self.pronunciation.spelled.device)

    @override
    def loss(self, head: nn.Module, output: Float[Tensor, "g w _"], tokens: Int[Tensor, "g w"]) -> Float[Tensor, ""]:
        spelled = self.pronunciation.phonemes(tokens)[..., : self.positions]
        return self.weight * Phonetic.loss(output.reshape(-1, len(PHONEMES)), spelled.flatten())

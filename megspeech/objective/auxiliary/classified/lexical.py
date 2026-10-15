"""The lexical auxiliary: each heard word's embedding classified into its word among every distinct word heard."""

from dataclasses import dataclass
from typing import override

from jaxtyping import Int
from torch import Tensor, nn

from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import Classified


@dataclass(frozen=True)
class LexicalTerm(Classified):
    """Every heard word's window, in or out of the 50, classified into its word among every distinct word heard
    (the `Lexicon`'s keys): the 60% of windows the heads never label get a word identity to discriminate."""

    weight: float
    lexicon: Lexicon

    @override
    def head(self, width: int) -> nn.Module:
        return nn.Linear(width, self.count()).to(self.lexicon.keys.device)

    @override
    def count(self) -> int:
        return len(self.lexicon.keys)

    @override
    def classify(self, token: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        return self.lexicon.rows(token)

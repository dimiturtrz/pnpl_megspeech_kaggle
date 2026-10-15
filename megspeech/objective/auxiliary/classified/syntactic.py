"""The syntactic auxiliary: each heard word's embedding classified into the part of speech of its text."""

from dataclasses import dataclass
from typing import override

from jaxtyping import Int
from torch import Tensor, nn

from megspeech.data.labels.syntax import TAGS, PartsOfSpeech
from megspeech.objective.term import Classified


@dataclass(frozen=True)
class SyntacticTerm(Classified):
    """Every heard word's window, in or out of the 50, classified into its universal part of speech: a placeholder
    class (noun, verb, ...) for the ~60% of windows the 50-way heads never label."""

    weight: float
    parts: PartsOfSpeech

    @override
    def head(self, width: int) -> nn.Module:
        return nn.Linear(width, self.count()).to(self.parts.tags.device)

    @override
    def count(self) -> int:
        return len(TAGS)

    @override
    def classify(self, token: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        return self.parts.classes(token)

"""The competition's word lists and how a heard word from the events files is spelled in them."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Int
from pnpl.competition import PRIMARY_VOCAB, SECONDARY_VOCAB

NON_WORD = re.compile(r"[^a-z’]")
ABSENT = -1


@dataclass(frozen=True)
class Vocabulary:
    """An ordered word list; a word's position is its label and its column in the submission."""

    words: tuple[str, ...]
    positions: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", {word: position for position, word in enumerate(self.words)})

    def __len__(self) -> int:
        return len(self.words)

    @staticmethod
    def spell(heard: str) -> str:
        """Lower-case, punctuation-free, with the curly apostrophe the vocabulary files use (`it’s`)."""
        return NON_WORD.sub("", heard.lower().replace("'", "’"))

    def labels(self, heard: Iterable[str]) -> Int[np.ndarray, " n"]:
        """Each heard word's position in this vocabulary, or `ABSENT`."""
        return np.array([self.positions.get(self.spell(word), ABSENT) for word in heard], dtype=np.int64)


PRIMARY = Vocabulary(tuple(PRIMARY_VOCAB))
SECONDARY = Vocabulary(tuple(SECONDARY_VOCAB))

"""The competition's word lists and how a heard word from the events files is spelled in them."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from jaxtyping import Int
from pnpl.competition import PRIMARY_VOCAB, SECONDARY_VOCAB

NON_WORD = re.compile(r"[^a-z’]")
ABSENT = -1
VOCAB_SIZE = 50
IGNORE = -100
"""The label of a padding row; `cross_entropy`'s default `ignore_index`."""


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
    def spelled(heard: pd.Series) -> pd.Series:
        """Lower-case, punctuation-free, with the curly apostrophe the vocabulary files use (`it’s`)."""
        return heard.astype(str).str.lower().str.replace("'", "’", regex=False).str.replace(NON_WORD, "", regex=True)

    @classmethod
    def spell(cls, heard: str) -> str:
        return cls.spelled(pd.Series([heard], dtype=object)).iloc[0]

    def labels(self, heard: Iterable[str]) -> Int[np.ndarray, " n"]:
        """Each heard word's position in this vocabulary, or `ABSENT`."""
        spelled = self.spelled(pd.Series(list(heard), dtype=object))
        return spelled.map(self.positions).fillna(ABSENT).to_numpy(dtype=np.int64)

    @staticmethod
    def token(words: pd.Series) -> Int[np.ndarray, " n"]:
        """Each word lower-cased and hashed: the key every per-word table shares with a split's `token`."""
        return pd.util.hash_array(words.astype(str).str.lower().to_numpy()).view(np.int64)


PRIMARY = Vocabulary(tuple(PRIMARY_VOCAB))
SECONDARY = Vocabulary(tuple(SECONDARY_VOCAB))

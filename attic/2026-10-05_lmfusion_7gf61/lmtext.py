"""The true sentence text around every word row of the peer protocol's dumps (megspeech-7gf.61): the words said before
it in its sentence and the next ones after, in the row order `peer.Dump` wrote, from the word tables alone (no MEG).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Int
from torch import Tensor

from megspeech.data.cache.run_cache import NO_SENTENCE
from megspeech.data.labels.vocabulary import ABSENT, IGNORE, PRIMARY
from megspeech.data.raw.runs import Run
from megspeech.evaluation.reader import Reader

Context = tuple[tuple[str, ...], tuple[str, ...]]
"""A row's left words and its next `RIGHT` words."""
RIGHT = 2
"""Following words kept for each row."""


@dataclass(frozen=True)
class Sentences:
    """Each row's word as spoken, its `left` words in the sentence and up to `RIGHT` words after it, in `Rows` order."""

    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def labels(words: pd.Series) -> Int[np.ndarray, " n"]:
        """The vocabulary label each spoken word carries as `Rows.y` does."""
        labels = PRIMARY.labels(words)
        return np.where(labels == ABSENT, IGNORE, labels)

    @classmethod
    def of(cls, reader: Reader, runs: Sequence[Run]) -> "Sentences":  # devtools-ignore: test-mirror
        """The words of `runs` as the scorer reads them, run after run (domain after domain is the same order)."""
        full, rows = reader.spoken(list(runs))
        spoken = full.assign(word=full["word"].astype(str)).groupby(["run", "sentence"], sort=False)["word"].apply(list)
        left: list[tuple[str, ...]] = []
        right: list[tuple[str, ...]] = []
        for run, sentence, position in zip(rows["run"], rows["sentence"], rows["position"], strict=True):
            words: list[str] = [] if sentence == NO_SENTENCE else spoken[(run, sentence)]
            at = int(position) if words else 0
            left.append(tuple(words[:at]))
            right.append(tuple(words[at + 1 : at + 1 + RIGHT]))
        around = pd.DataFrame({"left": pd.Series(left, dtype=object), "right": pd.Series(right, dtype=object)})
        named = rows[["run", "word"]].reset_index(drop=True).assign(word=lambda frame: frame["word"].astype(str))
        return cls(pd.concat([named, around], axis=1))

    def agrees(self, y: Int[np.ndarray, " n"]) -> float:
        """The share of rows whose spoken word carries the label `y` records for it."""
        return float((self.labels(self.frame["word"]) == y).mean())

    @classmethod
    def joined(cls, parts: Sequence["Sentences"]) -> "Sentences":
        """`parts` one after another."""
        return cls(pd.concat([part.frame for part in parts], ignore_index=True))

    def contexts(self, take: Int[Tensor, " k"]) -> list[Context]:
        """The left words and following words of the `take` rows."""
        left, right = self.frame["left"], self.frame["right"]
        return [(left.iloc[i], right.iloc[i]) for i in take.tolist()]

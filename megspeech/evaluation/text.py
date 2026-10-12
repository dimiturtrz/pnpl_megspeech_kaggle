"""The words a run was heard saying, keyed so two runs of one text match word for word."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Int

from megspeech.data.raw.runs import SHERLOCK, Column

RULER = SHERLOCK
"""The validation task whose text the pooled runs repeat."""


@dataclass(frozen=True)
class Text:
    """One run's heard words (a word table with `sentence`, `position`, `word`, `onset`), keyed word by word: by its
    sentence, its place in that sentence and its lower-cased spelling, so two runs of one text match word for word."""

    words: pd.DataFrame

    @classmethod
    def of(cls, table: pd.DataFrame) -> "Text":
        """`table`'s words with their `position` in their sentence of their run."""
        return cls(table.assign(position=table.groupby([Column.RUN, Column.SENTENCE]).cumcount()))

    @property
    def keys(self) -> pd.MultiIndex:
        spelled = self.words[Column.WORD].astype(str).str.lower()
        return pd.MultiIndex.from_arrays([self.words[Column.SENTENCE], self.words[Column.POSITION], spelled])

    def matches(self, other: "Text") -> Int[np.ndarray, " n"]:
        """Each word's row in `other` with the same key, -1 where `other` has none."""
        return other.keys.get_indexer(self.keys)

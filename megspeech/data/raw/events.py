"""A run's `events.tsv`: pnpl's schema, which every raw source here is written into and every reader reads."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from megspeech.data.constants import MISSING


class Event(StrEnum):
    """A column of a run's `events.tsv`: pnpl's schema, which every raw source here is written into."""

    KIND = "kind"
    SEGMENT = "segment"
    SENTENCE = "sentenceidx"
    WORD = "wordidx"
    PHONEME = "phonemeidx"
    ONSET = "timemeg"
    DURATION = "duration"
    CHAPTER = "timechapter"


class Kind(StrEnum):
    """What an `events.tsv` row marks: a heard word or one of its phonemes."""

    WORD = "word"
    PHONEME = "phoneme"


@dataclass(frozen=True)
class Events:
    """One run's `events.tsv`, read with only an empty cell missing (`MISSING`)."""

    path: Path

    def read(self, *columns: Event) -> pd.DataFrame:
        """Every row's `kind`, `segment` and `columns`."""
        return pd.read_csv(
            self.path,
            sep="\t",
            usecols=list(dict.fromkeys([Event.KIND, Event.SEGMENT, *columns])),
            keep_default_na=False,
            na_values=MISSING,
        )

    @staticmethod
    def heard(table: pd.DataFrame) -> pd.Series:
        """Which rows are heard words: `kind` word with a `segment` spoken."""
        return (table[Event.KIND] == Kind.WORD) & table[Event.SEGMENT].notna()

    def words(self, *columns: Event) -> pd.DataFrame:
        """The heard words' rows."""
        table = self.read(*columns)
        return table[self.heard(table)]

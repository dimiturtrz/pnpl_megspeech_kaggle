"""Praat TextGrid interval tiers, in the long text form forced aligners write."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

TIER = re.compile(r'class = "IntervalTier"\s+name = "(?P<name>[^"]*)"(?P<body>.*?)(?=\n\s*item \[|\Z)', re.DOTALL)
INTERVAL = re.compile(r'xmin = (?P<start>[\d.eE+-]+)\s+xmax = (?P<end>[\d.eE+-]+)\s+text = "(?P<text>[^"]*)"')


class Interval(StrEnum):
    """A tier table's column: an interval's bounds in seconds and its label."""

    START = "start"
    END = "end"
    TEXT = "text"


class Tier(StrEnum):
    """The tiers a forced aligner writes: one interval per word, one per phone."""

    WORDS = "words"
    PHONES = "phones"


@dataclass(frozen=True)
class TextGrid:
    """A TextGrid's interval tiers by name, each a table of `Interval` columns."""

    tiers: dict[str, pd.DataFrame]

    @classmethod
    def read(cls, path: Path) -> "TextGrid":
        tiers = {
            tier["name"]: pd.DataFrame(
                [
                    (float(m[Interval.START]), float(m[Interval.END]), m[Interval.TEXT])
                    for m in INTERVAL.finditer(tier["body"])
                ],
                columns=list(Interval),
            )
            for tier in TIER.finditer(path.read_text(encoding="utf-8"))
        }
        return cls(tiers)

    @classmethod
    def aligned(cls, words: pd.DataFrame, phones: pd.DataFrame) -> "TextGrid":
        """A forced alignment's two tiers and nothing else."""
        return cls({Tier.WORDS: words, Tier.PHONES: phones})

    @property
    def words(self) -> pd.DataFrame:
        return self.tiers[Tier.WORDS]

    @property
    def phones(self) -> pd.DataFrame:
        return self.tiers[Tier.PHONES]

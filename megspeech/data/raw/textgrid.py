"""Praat TextGrid interval tiers, in the long text form forced aligners write."""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TIER = re.compile(r'class = "IntervalTier"\s+name = "(?P<name>[^"]*)"(?P<body>.*?)(?=\n\s*item \[|\Z)', re.DOTALL)
INTERVAL = re.compile(r'xmin = (?P<start>[\d.eE+-]+)\s+xmax = (?P<end>[\d.eE+-]+)\s+text = "(?P<text>[^"]*)"')


@dataclass(frozen=True)
class TextGrid:
    """A TextGrid's interval tiers by name, each a table of `start`, `end` (seconds) and `text`."""

    tiers: dict[str, pd.DataFrame]

    @classmethod
    def read(cls, path: Path) -> "TextGrid":
        tiers = {
            tier["name"]: pd.DataFrame(
                [(float(m["start"]), float(m["end"]), m["text"]) for m in INTERVAL.finditer(tier["body"])],
                columns=["start", "end", "text"],
            )
            for tier in TIER.finditer(path.read_text(encoding="utf-8"))
        }
        return cls(tiers)

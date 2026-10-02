"""Spans of words heard together: the unit the context model reads.

A holdout sentence is a run of vocabulary words only — onset gaps p50 0.21 s, max 0.92 s, at most nine
words. Training spans are cut to match: consecutive vocabulary words of one run whose onset gaps stay within
the holdout's widest gap, chunked to nine.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Float, Int

MAX_WORDS = 9
"""Width of the holdout's `word_onsets_s`."""

MAX_GAP = 0.92
"""Widest within-sentence onset gap in the Deep holdout (subject 0, 156 sentences)."""

PAD = -1


@dataclass(frozen=True)
class Spans:
    """Index rows into a flat word array, `PAD`-padded, with each word's onset relative to its span's first."""

    index: Int[np.ndarray, "g w"]
    onset: Float[np.ndarray, "g w"]

    @classmethod
    def assemble(
        cls, span: Int[np.ndarray, " n"], slot: Int[np.ndarray, " n"], onset: Float[np.ndarray, " n"]
    ) -> "Spans":
        index = np.full((span.max(initial=-1) + 1, MAX_WORDS), PAD, dtype=np.int64)
        index[span, slot] = np.arange(len(span))
        relative = np.zeros(index.shape)
        first = onset[index[:, 0]]
        relative[span, slot] = onset - first[span]
        return cls(index, relative)

    @classmethod
    def by_gap(cls, onset: Float[np.ndarray, " n"], max_gap: float = MAX_GAP) -> "Spans":
        """Split where the next onset is too far ahead or goes back (a new run), then chunk to `MAX_WORDS`."""
        gap = np.diff(onset)
        new = np.r_[True, (gap < 0) | (gap > max_gap)]
        segment = np.cumsum(new) - 1
        position = np.arange(len(onset)) - np.flatnonzero(new)[segment]
        span = np.cumsum(position % MAX_WORDS == 0) - 1
        return cls.assemble(span, position % MAX_WORDS, onset)

    @classmethod
    def by_key(cls, keys: pd.DataFrame, onset: Float[np.ndarray, " n"]) -> "Spans":
        """One span per distinct key row (e.g. a holdout sentence), in the order the words appear."""
        span = keys.groupby(list(keys.columns), sort=False).ngroup().to_numpy()
        slot = keys.groupby(list(keys.columns), sort=False).cumcount().to_numpy()
        return cls.assemble(span, slot, onset)

    @property
    def lengths(self) -> Int[np.ndarray, " g"]:
        return (self.index != PAD).sum(axis=1)

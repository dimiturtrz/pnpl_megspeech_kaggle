"""Spans read as text: each span's words as their token sequence, the sentence it recites."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from megspeech.data.spans.spans import PAD

RECITED = 3
"""Spans this long recur only when a sentence is replayed; 2-word pairs of 50 words (2500) recur by chance."""


@dataclass(frozen=True)
class SpanText:
    """Span `g`'s words are `index[g]` (`PAD` past its end); word `i` reads as `token[i]`."""

    index: Int[np.ndarray, "g w"]
    token: Int[np.ndarray, " n"]

    @property
    def valid(self) -> Bool[np.ndarray, "g w"]:
        return self.index != PAD

    @property
    def lengths(self) -> Int[np.ndarray, " g"]:
        return self.valid.sum(axis=1)

    def sequences(self) -> list[tuple[int, ...]]:
        """Every span's token sequence, in span order."""
        return [tuple(self.token[row[row != PAD]].tolist()) for row in self.index]

    def repeats(self, power: float) -> Float[np.ndarray, " n"]:
        """Each word's weight from how often its span's text recurs among the spans, to the `-power`; words of spans
        shorter than `RECITED`, and words in no span, keep 1."""
        packed = np.take_along_axis(self.index, np.argsort(~self.valid, axis=1, kind="stable"), axis=1)
        valid = packed != PAD
        keyed = np.c_[valid.sum(axis=1), np.where(valid, self.token[packed.clip(min=0)], 0)]
        _, inverse, counts = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
        recited = valid.sum(axis=1) >= RECITED
        span = np.where(recited, counts[inverse.reshape(-1)].astype(np.float64) ** -power, 1.0)
        scale = np.ones(len(self.token))
        scale[packed[valid]] = np.broadcast_to(span[:, None], packed.shape)[valid]
        return scale

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[np.ndarray, " k"]:
        """Words of multi-word spans whose text is not in `seen`, span by span."""
        unheard = np.array([sequence not in seen for sequence in self.sequences()], dtype=bool)
        rows = self.index[(self.lengths > 1) & unheard]
        return rows[rows != PAD]

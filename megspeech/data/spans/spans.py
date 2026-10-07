"""Spans of words heard together: the unit the context model reads.

A Kaggle test sentence carries every word's onset, out-of-vocabulary ones included — onset gaps p50 0.20 s, at most
nine words (`word_onsets_s` is `(N_sent, 9)` in `pnpl.competition.holdout`). Training spans are cut one of two
ways: by sentence, every word of one PNPL sentence (`sentenceidx`, as pnpl's `LibriBrainSentence` groups them), a
sentence longer than nine split into the fewest near-equal pieces; or, the older gap rule, consecutive vocabulary
words of one run whose onset gaps stay within `MAX_GAP`, chunked to nine.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Float, Int

MAX_WORDS = 9
"""Width of the Kaggle test's `word_onsets_s`."""

MAX_GAP = 0.92
"""Widest within-sentence onset gap in the Deep Kaggle test (subject 0, 156 sentences)."""

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
    def by_sentence(
        cls, group: Int[np.ndarray, " n"], onset: Float[np.ndarray, " n"], words: float = MAX_WORDS
    ) -> "Spans":
        """One span per run of equal consecutive `group` codes, cut into the more of `ceil(n / MAX_WORDS)` and
        `round(n / words)` near-equal pieces: `words` the length a piece aims at, at `MAX_WORDS` the fewest.

        Equal pieces rather than nines-then-a-remainder: a 10-word sentence is 5 + 5, not 9 + 1, so no span
        loses its context to a cut.
        """
        span, slot = cls.piece_codes(group, words)
        return cls.assemble(span, slot, onset)

    @staticmethod
    def piece_codes(
        group: Int[np.ndarray, " n"], words: float = MAX_WORDS
    ) -> tuple[Int[np.ndarray, " n"], Int[np.ndarray, " n"]]:
        """Each word's piece (unique across `group`) and its slot in it, as `by_sentence` cuts them."""
        new = np.r_[True, group[1:] != group[:-1]][: len(group)]
        sentence = np.cumsum(new) - 1
        first = np.flatnonzero(new)
        sizes = np.diff(np.r_[first, len(group)])
        cuts = np.maximum(-(-sizes // MAX_WORDS), np.round(sizes / words).astype(np.int64))
        size, pieces = sizes[sentence], cuts[sentence]
        position = np.arange(len(group)) - first[sentence]
        piece = position * pieces // size
        start = -(-piece * size // pieces)
        return np.r_[0, np.cumsum(cuts)][sentence] + piece, position - start

    @classmethod
    def by_pieces(
        cls, group: Int[np.ndarray, " n"], onset: Float[np.ndarray, " n"], generator: np.random.Generator
    ) -> "Spans":
        """One span per piece of a run of equal consecutive `group` codes, piece lengths i.i.d. uniform on
        `1..MAX_WORDS`; the last piece of a run takes what remains."""
        new = np.r_[True, group[1:] != group[:-1]][: len(group)]
        sentence = np.cumsum(new) - 1
        first = np.flatnonzero(new)[sentence]
        end = first + np.diff(np.r_[np.flatnonzero(new), len(group)])[sentence]
        draw = generator.integers(1, MAX_WORDS + 1, len(group))
        total = np.cumsum(draw)
        base = (total - draw)[first]
        stops = np.minimum(first + total - base, end)
        word = np.arange(len(group))
        piece = np.unique(np.searchsorted(stops, word, side="right"), return_inverse=True)[1]
        start = np.flatnonzero(np.r_[True, piece[1:] != piece[:-1]][: len(group)])
        return cls.assemble(piece, word - start[piece], onset)

    @classmethod
    def by_key(cls, keys: pd.DataFrame, onset: Float[np.ndarray, " n"]) -> "Spans":
        """One span per distinct key row (e.g. a Kaggle test sentence), in the order the words appear."""
        grouped = keys.groupby(list(keys.columns), sort=False)
        return cls.assemble(grouped.ngroup().to_numpy(), grouped.cumcount().to_numpy(), onset)

    @property
    def lengths(self) -> Int[np.ndarray, " g"]:
        return (self.index != PAD).sum(axis=1)

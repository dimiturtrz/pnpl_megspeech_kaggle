"""Spans of words heard together: the unit the context model reads.

A holdout sentence is a run of vocabulary words only — onset gaps p50 0.21 s, max 0.92 s, at most nine
words (`word_onsets_s` is `(N_sent, 9)` in `pnpl.competition.holdout`). Training spans are cut one of two ways:
by gap, consecutive vocabulary words of one run whose onset gaps stay within the holdout's widest gap, chunked
to nine; or by sentence, every word of one PNPL sentence (`sentenceidx`, as pnpl's `LibriBrainSentence`
groups them), a sentence longer than nine split into the fewest near-equal pieces.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int

from megspeech.data.labels.vocabulary import IGNORE

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
    def by_sentence(cls, group: Int[np.ndarray, " n"], onset: Float[np.ndarray, " n"]) -> "Spans":
        """One span per run of equal consecutive `group` codes, cut into `ceil(n / MAX_WORDS)` near-equal pieces.

        Equal pieces rather than nines-then-a-remainder: a 10-word sentence is 5 + 5, not 9 + 1, so no span
        loses its context to a cut.
        """
        new = np.r_[True, group[1:] != group[:-1]][: len(group)]
        sentence = np.cumsum(new) - 1
        first = np.flatnonzero(new)
        sizes = np.diff(np.r_[first, len(group)])
        cuts = -(-sizes // MAX_WORDS)
        size, pieces = sizes[sentence], cuts[sentence]
        position = np.arange(len(group)) - first[sentence]
        piece = position * pieces // size
        start = -(-piece * size // pieces)
        return cls.assemble(np.r_[0, np.cumsum(cuts)][sentence] + piece, position - start, onset)

    @classmethod
    def by_key(cls, keys: pd.DataFrame, onset: Float[np.ndarray, " n"]) -> "Spans":
        """One span per distinct key row (e.g. a holdout sentence), in the order the words appear."""
        grouped = keys.groupby(list(keys.columns), sort=False)
        return cls.assemble(grouped.ngroup().to_numpy(), grouped.cumcount().to_numpy(), onset)

    @property
    def lengths(self) -> Int[np.ndarray, " g"]:
        return (self.index != PAD).sum(axis=1)


@dataclass(frozen=True)
class Rule:
    """How a grouping cuts: the word-table column, in samples into the run, at which it stops a word's window, the
    spans it cuts a word table into, and whether it carries words outside the vocabulary."""

    stop: str
    cut: Callable[[pd.DataFrame], Spans]
    unlabelled: bool


class Grouping(StrEnum):
    """`GAP`: vocabulary words only, spans by onset gap. `SENTENCE`: every word, spans by PNPL sentence.

    Under `SENTENCE` an out-of-vocabulary word still carries its MEG into its span's context, labelled `IGNORE`
    so no loss or score counts it.
    """

    GAP = "gap"
    SENTENCE = "sentence"

    @property
    def rule(self) -> Rule:
        return RULES[self]

    def keep(self, target: Int[np.ndarray, " n"]) -> Bool[np.ndarray, " n"]:
        """Of words with these targets (`IGNORE` outside the vocabulary), the ones this grouping carries."""
        return (target != IGNORE) | self.rule.unlabelled

    def spans(self, words: pd.DataFrame) -> Spans:
        """Spans over `words` (a word table in run order: `run`, `sentence`, `onset`)."""
        return self.rule.cut(words)

    def bound(self, words: pd.DataFrame) -> pd.Series:
        """Each of `words`' exclusive end sample in the loaded signal (`DeviceRecording.load`'s columns): its
        sentence epoch's end under `SENTENCE`, as the holdout cuts it; its run's end under `GAP`."""
        return words["start"] + words[self.rule.stop]


RULES = {
    Grouping.GAP: Rule("samples", lambda words: Spans.by_gap(words["onset"].to_numpy()), unlabelled=False),
    Grouping.SENTENCE: Rule(
        "stop",
        lambda words: Spans.by_sentence(
            words.groupby(["run", "sentence"], sort=False).ngroup().to_numpy(), words["onset"].to_numpy()
        ),
        unlabelled=True,
    ),
}

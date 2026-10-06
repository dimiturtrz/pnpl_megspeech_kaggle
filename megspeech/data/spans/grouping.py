"""How a training run groups words into spans."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int

from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import RATE
from megspeech.data.labels.vocabulary import ABSENT, IGNORE, PRIMARY, Vocabulary
from megspeech.data.spans.pieces import Pieces
from megspeech.data.spans.sentences import Sentences
from megspeech.data.spans.spans import Spans

KAGGLE_WORDS = 5.56
"""Mean words per Deep Kaggle test sentence epoch (`word_onsets_s` mask, subject 0): pieces this long match its
spans, where whole PNPL sentences run 6.8-7.4 words in validation."""


@dataclass(frozen=True)
class Rule:
    """How a grouping cuts: the word-table column, in samples into the run, at which it stops a word's window, the
    spans it cuts a word table into, and whether it carries words outside the vocabulary."""

    stop: str
    cut: Callable[[pd.DataFrame], Spans]
    unlabelled: bool


class Grouping(StrEnum):
    """`GAP`: vocabulary words only, spans by onset gap. `SENTENCE`: every word, spans by PNPL sentence. `KAGGLE`:
    every word, each PNPL sentence cut into pieces of the Kaggle test's length (`KAGGLE_WORDS`). `SPEC`:
    every word, each PNPL sentence cut into pieces of random length uniform on 1 to `MAX_WORDS`, seeded by `SEED`.

    Under `SENTENCE`, `KAGGLE` and `SPEC` an out-of-vocabulary word still carries its MEG into its span's context,
    labelled `IGNORE` so no loss or score counts it.
    """

    GAP = "gap"
    SENTENCE = "sentence"
    KAGGLE = "kaggle"
    SPEC = "spec"

    @property
    def rule(self) -> Rule:
        return RULES[self]

    def keep(self, target: Int[np.ndarray, " n"]) -> Bool[np.ndarray, " n"]:
        """Of words with these targets (`IGNORE` outside the vocabulary), the ones this grouping carries."""
        return (target != IGNORE) | self.rule.unlabelled

    def carried(self, table: pd.DataFrame, rate: float = RATE) -> pd.DataFrame:
        """The words of `table` (sampled at `rate`) this grouping carries and whose 1 s window fits in their run,
        with their `target` and `token` (the lower-cased word, hashed)."""
        labels = PRIMARY.labels(table["word"])
        target = np.where(labels == ABSENT, IGNORE, labels)
        return table.assign(target=target, token=Vocabulary.token(table["word"]))[
            self.keep(target) & RunCache.inside(table, round(rate))
        ]

    def spans(self, words: pd.DataFrame) -> Spans:
        """Spans over `words` (a word table in run order: `run`, `sentence`, `onset`)."""
        return self.rule.cut(words)

    def bound(self, words: pd.DataFrame) -> pd.Series:
        """Each of `words`' exclusive end sample in the loaded signal (`DeviceRecording.load`'s columns): its
        sentence epoch's end under `SENTENCE`, as the Kaggle test cuts it; its run's end under `GAP`."""
        return words["start"] + words[self.rule.stop]


RULES = {
    Grouping.GAP: Rule("samples", lambda words: Spans.by_gap(words["onset"].to_numpy()), unlabelled=False),
    Grouping.SENTENCE: Rule("stop", Sentences(), unlabelled=True),
    Grouping.KAGGLE: Rule("stop", Sentences(KAGGLE_WORDS), unlabelled=True),
    Grouping.SPEC: Rule("stop", Pieces(), unlabelled=True),
}

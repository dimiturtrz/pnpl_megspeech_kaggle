"""Which heard words a split carries, and how they group into spans (see `megspeech.data.spans.spans`)."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans


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

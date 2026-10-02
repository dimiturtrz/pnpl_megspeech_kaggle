"""Which heard words a split carries, and how they group into spans (see `megspeech.data.spans`)."""

from enum import StrEnum

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int

from megspeech.data.spans import Spans
from megspeech.data.vocabulary import IGNORE


class Grouping(StrEnum):
    """`GAP`: vocabulary words only, spans by onset gap. `SENTENCE`: every word, spans by PNPL sentence.

    Under `SENTENCE` an out-of-vocabulary word still carries its MEG into its span's context, labelled `IGNORE`
    so no loss or score counts it.
    """

    GAP = "gap"
    SENTENCE = "sentence"

    def keep(self, target: Int[np.ndarray, " n"]) -> Bool[np.ndarray, " n"]:
        """Of words with these targets (`IGNORE` outside the vocabulary), the ones this grouping carries."""
        return (target != IGNORE) | (self is Grouping.SENTENCE)

    def spans(self, words: pd.DataFrame) -> Spans:
        """Spans over `words` (a word table in run order: `run`, `sentence`, `onset`)."""
        onset = words["onset"].to_numpy()
        if self is Grouping.GAP:
            return Spans.by_gap(onset)
        sentence = words.groupby(["run", "sentence"], sort=False).ngroup().to_numpy()
        return Spans.by_sentence(sentence, onset)

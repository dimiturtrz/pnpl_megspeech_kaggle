"""The sentence cut of a word table."""

import pandas as pd

from megspeech.data.spans.spans import MAX_WORDS, Spans


class Sentences:
    """The sentence cut, pieces aiming at `words` long."""

    def __init__(self, words: float = MAX_WORDS) -> None:
        self.words = words

    def __call__(self, table: pd.DataFrame) -> Spans:
        group = table.groupby(["run", "sentence"], sort=False).ngroup().to_numpy()
        return Spans.by_sentence(group, table["onset"].to_numpy(), self.words)

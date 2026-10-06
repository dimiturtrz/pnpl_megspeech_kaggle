"""The random cut of a word table."""

import numpy as np
import pandas as pd

from megspeech.data.spans.spans import Spans

SEED = 0
"""Seed of the piece lengths `Spans.by_pieces` draws: one table always cuts the same way."""


class Pieces:
    """The random cut: pieces of uniform length on `1..MAX_WORDS`, drawn from a generator seeded by `SEED`."""

    def __call__(self, table: pd.DataFrame) -> Spans:
        group = table.groupby(["run", "sentence"], sort=False).ngroup().to_numpy()
        return Spans.by_pieces(group, table["onset"].to_numpy(), np.random.default_rng(SEED))

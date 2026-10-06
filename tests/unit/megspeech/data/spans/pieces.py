import numpy as np
import pandas as pd

from megspeech.data.spans.grouping import Grouping
from megspeech.data.spans.pieces import Pieces
from megspeech.data.spans.spans import MAX_WORDS


def test_pieces() -> None:
    long = pd.DataFrame({"run": ["a"] * 40, "sentence": [0] * 40, "onset": np.arange(40) * 0.3})
    spans = Pieces()(long)
    assert spans.lengths.sum() == 40
    assert spans.lengths.max() <= MAX_WORDS
    assert (Grouping.SPEC.spans(long).index == spans.index).all()
    assert Grouping.SPEC.rule.unlabelled

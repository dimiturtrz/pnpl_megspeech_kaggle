import numpy as np
import pandas as pd

from megspeech.data.grouping import Grouping
from megspeech.data.spans import PAD
from megspeech.data.vocabulary import IGNORE


def test_keep() -> None:
    target = np.array([3, IGNORE, 0])
    assert Grouping.GAP.keep(target).tolist() == [True, False, True]
    assert Grouping.SENTENCE.keep(target).tolist() == [True, True, True]


def test_spans() -> None:
    words = pd.DataFrame({"run": ["a", "a", "a", "b"], "sentence": [0, 0, 1, 0], "onset": [0.0, 0.2, 0.4, 0.1]})
    assert Grouping.SENTENCE.spans(words).index[:, :2].tolist() == [[0, 1], [2, PAD], [3, PAD]]
    assert Grouping.GAP.spans(words).index[:, :3].tolist() == [[0, 1, 2], [3, PAD, PAD]]


def test_bound() -> None:
    words = pd.DataFrame({"start": [0, 100], "samples": [80, 80], "stop": [60, 70]})
    assert Grouping.SENTENCE.bound(words).tolist() == [60, 170]
    assert Grouping.GAP.bound(words).tolist() == [80, 180]

import numpy as np
import pandas as pd

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.grouping import Grouping
from megspeech.data.spans.spans import PAD

WORDS = pd.DataFrame({"run": ["a", "a", "a", "b"], "sentence": [0, 0, 1, 0], "onset": [0.0, 0.2, 0.4, 0.1]})


def test_rule() -> None:
    assert (Grouping.GAP.rule.stop, Grouping.GAP.rule.unlabelled) == ("samples", False)
    assert (Grouping.SENTENCE.rule.stop, Grouping.SENTENCE.rule.unlabelled) == ("stop", True)
    assert Grouping.GAP.rule.cut(WORDS).lengths.tolist() == [3, 1]


def test_keep() -> None:
    target = np.array([3, IGNORE, 0])
    assert Grouping.GAP.keep(target).tolist() == [True, False, True]
    assert Grouping.SENTENCE.keep(target).tolist() == [True, True, True]


def test_spans() -> None:
    assert Grouping.SENTENCE.spans(WORDS).index[:, :2].tolist() == [[0, 1], [2, PAD], [3, PAD]]
    assert Grouping.GAP.spans(WORDS).index[:, :3].tolist() == [[0, 1, 2], [3, PAD, PAD]]
    nine = pd.DataFrame({"run": ["a"] * 9, "sentence": [0] * 9, "onset": np.arange(9) * 0.3})
    assert Grouping.KAGGLE.spans(nine).lengths.tolist() == [5, 4]
    assert Grouping.SENTENCE.spans(nine).lengths.tolist() == [9]


def test_bound() -> None:
    words = pd.DataFrame({"start": [0, 100], "samples": [80, 80], "stop": [60, 70]})
    assert Grouping.SENTENCE.bound(words).tolist() == [60, 170]
    assert Grouping.GAP.bound(words).tolist() == [80, 180]
    pieced = words.assign(piece_stop=[40, 50])
    assert Grouping.KAGGLE.bound(pieced).tolist() == [40, 150]
    assert Grouping.SENTENCE.bound(pieced).tolist() == [60, 170]


def test_opening() -> None:
    words = pd.DataFrame({"start": [0, 100], "opening": [5, 6], "piece_opening": [7, 8]})
    assert Grouping.SENTENCE.opening(words).tolist() == [5, 106]
    assert Grouping.KAGGLE.opening(words).tolist() == [7, 108]


def test_following() -> None:
    words = pd.DataFrame({"start": [0, 100], "following": [5, 6], "piece_following": [7, 8]})
    assert Grouping.SENTENCE.following(words).tolist() == [5, 106]
    assert Grouping.KAGGLE.following(words).tolist() == [7, 108]

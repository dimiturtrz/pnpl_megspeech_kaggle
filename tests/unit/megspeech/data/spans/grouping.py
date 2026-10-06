from pathlib import Path

import numpy as np
import pandas as pd

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.grouping import Grouping
from megspeech.data.spans.spans import PAD
from tests.unit.megspeech.data.corpus import THE, A, source

WORDS = pd.DataFrame({"run": ["a", "a", "a", "b"], "sentence": [0, 0, 1, 0], "onset": [0.0, 0.2, 0.4, 0.1]})


def test_rule() -> None:
    assert (Grouping.GAP.rule.stop, Grouping.GAP.rule.unlabelled) == ("samples", False)
    assert (Grouping.SENTENCE.rule.stop, Grouping.SENTENCE.rule.unlabelled) == ("stop", True)
    assert Grouping.GAP.rule.cut(WORDS).lengths.tolist() == [3, 1]


def test_keep() -> None:
    target = np.array([3, IGNORE, 0])
    assert Grouping.GAP.keep(target).tolist() == [True, False, True]
    assert Grouping.SENTENCE.keep(target).tolist() == [True, True, True]


def test_carried(tmp_path: Path) -> None:
    table = source(tmp_path).words
    sentence = Grouping.SENTENCE.carried(table)
    assert sentence["word"].tolist() == ["the", "zebra", "a", "the", "a"]
    assert sentence["target"].tolist() == [THE, IGNORE, A, THE, A]
    assert Grouping.GAP.carried(table)["word"].tolist() == ["the", "a", "the", "a"]


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

import numpy as np
import pandas as pd

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import MAX_WORDS, PAD, Grouping, Spans


def test_by_gap() -> None:
    onset = np.array([0.0, 0.2, 0.4, 2.0, 2.3, 0.1])
    spans = Spans.by_gap(onset)
    assert spans.index[:, :3].tolist() == [[0, 1, 2], [3, 4, PAD], [5, PAD, PAD]]
    assert np.allclose(spans.onset[1, :2], [0.0, 0.3])


def test_by_gap_chunks_long_runs() -> None:
    spans = Spans.by_gap(np.arange(MAX_WORDS + 2) * 0.1)
    assert spans.lengths.tolist() == [MAX_WORDS, 2]
    assert np.allclose(spans.onset[1, :2], [0.0, 0.1])


def test_by_sentence() -> None:
    group = np.r_[[0] * 3, [1] * 10, [2] * 19, [5]]
    spans = Spans.by_sentence(group, np.arange(len(group)) * 0.5)
    assert spans.lengths.tolist() == [3, 5, 5, 7, 6, 6, 1]
    assert spans.index[spans.index != PAD].tolist() == list(range(len(group)))
    assert spans.index[1, :5].tolist() == [3, 4, 5, 6, 7]
    assert np.allclose(spans.onset[2, :5], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert Spans.by_sentence(np.array([], np.int64), np.array([])).index.shape == (0, MAX_WORDS)


def test_by_key() -> None:
    keys = pd.DataFrame({"source": ["sentence"] * 3 + ["word"], "group": [0, 0, 1, -4]})
    spans = Spans.by_key(keys, np.array([0.0, 0.2, 0.0, 0.0]))
    assert spans.index[:, :2].tolist() == [[0, 1], [2, PAD], [3, PAD]]
    assert np.allclose(spans.onset[0, :2], [0.0, 0.2])


def test_assemble() -> None:
    spans = Spans.assemble(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1.0, 1.5, 3.0]))
    assert spans.index.shape == (2, MAX_WORDS)
    assert np.allclose(spans.onset[:, :2], [[0.0, 0.5], [0.0, 0.0]])


def test_lengths() -> None:
    assert Spans.by_gap(np.array([0.0, 5.0, 5.1])).lengths.tolist() == [1, 2]


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


def test_bound() -> None:
    words = pd.DataFrame({"start": [0, 100], "samples": [80, 80], "stop": [60, 70]})
    assert Grouping.SENTENCE.bound(words).tolist() == [60, 170]
    assert Grouping.GAP.bound(words).tolist() == [80, 180]

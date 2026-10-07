import numpy as np
import pandas as pd

from megspeech.data.spans.grouping import KAGGLE_WORDS
from megspeech.data.spans.spans import MAX_WORDS, PAD, Spans


def test_by_gap() -> None:
    onset = np.array([0.0, 0.2, 0.4, 2.0, 2.3, 0.1])
    spans = Spans.by_gap(onset)
    assert spans.index[:, :3].tolist() == [[0, 1, 2], [3, 4, PAD], [5, PAD, PAD]]
    assert np.allclose(spans.onset[1, :2], [0.0, 0.3])


def test_by_gap_chunks_long_runs() -> None:
    spans = Spans.by_gap(np.arange(MAX_WORDS + 2) * 0.1)
    assert spans.lengths.tolist() == [MAX_WORDS, 2]
    assert np.allclose(spans.onset[1, :2], [0.0, 0.1])


def test_piece_codes() -> None:
    group = np.r_[[0] * 3, [1] * 10, [5]]
    piece, slot = Spans.piece_codes(group)
    assert piece.tolist() == [0] * 3 + [1] * 5 + [2] * 5 + [3]
    assert slot.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]
    assert Spans.piece_codes(group, 5.56)[0].tolist() == piece.tolist()


def test_by_sentence() -> None:
    group = np.r_[[0] * 3, [1] * 10, [2] * 19, [5]]
    spans = Spans.by_sentence(group, np.arange(len(group)) * 0.5)
    assert spans.lengths.tolist() == [3, 5, 5, 7, 6, 6, 1]
    assert spans.index[spans.index != PAD].tolist() == list(range(len(group)))
    assert spans.index[1, :5].tolist() == [3, 4, 5, 6, 7]
    assert np.allclose(spans.onset[2, :5], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert Spans.by_sentence(np.array([], np.int64), np.array([])).index.shape == (0, MAX_WORDS)


def test_by_sentence_aims_at_words() -> None:
    """Aiming at the Kaggle test's length cuts a nine-word sentence in two and a 17-word one in three, but leaves a
    seven-word one whole; aiming at nine is the fewest-pieces cut."""
    group = np.r_[[0] * 9, [1] * 7, [2] * 17]
    onset = np.arange(len(group)) * 0.5
    assert Spans.by_sentence(group, onset, KAGGLE_WORDS).lengths.tolist() == [5, 4, 7, 6, 6, 5]
    assert Spans.by_sentence(group, onset, MAX_WORDS).lengths.tolist() == [9, 7, 9, 8]


def test_by_pieces() -> None:
    sizes = [3, MAX_WORDS, 40] * 30
    group = np.repeat(np.arange(len(sizes)), sizes)
    onset = np.arange(len(group)) * 0.5
    spans = Spans.by_pieces(group, onset, np.random.default_rng(0))
    assert spans.index[spans.index != PAD].tolist() == list(range(len(group)))
    assert spans.lengths.min() >= 1
    assert spans.lengths.max() <= MAX_WORDS
    assert (np.diff(spans.index[:, 0]) > 0).all()
    covered = np.where(spans.index == PAD, spans.index[:, :1], spans.index)
    assert (group[covered].max(axis=1) == group[covered].min(axis=1)).all()
    assert np.allclose(spans.onset[:, 0], 0.0)
    again = Spans.by_pieces(group, onset, np.random.default_rng(0))
    assert (again.index == spans.index).all()
    assert 4.0 < spans.lengths.mean() < 6.0
    short = Spans.by_pieces(np.zeros(3, np.int64), np.arange(3.0), np.random.default_rng(0))
    assert short.lengths.sum() == 3
    assert Spans.by_pieces(np.array([], np.int64), np.array([]), np.random.default_rng(0)).index.shape == (0, MAX_WORDS)


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

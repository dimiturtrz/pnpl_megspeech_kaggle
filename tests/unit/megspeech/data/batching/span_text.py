import numpy as np

from megspeech.data.batching.span_text import SpanText
from megspeech.data.spans.spans import PAD, Spans

TOKEN = np.array([5, 6, 7, 5, 6, 7, 9, 9])
INDEX = np.array([[0, 1, 2], [3, 4, 5], [6, 7, PAD]])


def test_valid() -> None:
    assert SpanText(INDEX, TOKEN).valid.tolist() == [[True] * 3, [True] * 3, [True, True, False]]


def test_lengths() -> None:
    assert SpanText(INDEX, TOKEN).lengths.tolist() == [3, 3, 2]


def test_sequences() -> None:
    assert SpanText(INDEX, TOKEN).sequences() == [(5, 6, 7), (5, 6, 7), (9, 9)]
    spans = Spans.by_gap(np.array([0.0, 0.2, 9.0]))
    assert set(SpanText(spans.index, np.array([4, 5, 6])).sequences()) == {(4, 5), (6,)}


def test_repeats() -> None:
    """A recited text heard twice weighs each of its words half at power 1; a span too short to recite keeps 1."""
    text = SpanText(INDEX, TOKEN)
    assert text.repeats(1.0).tolist() == [0.5] * 6 + [1.0, 1.0]
    assert text.repeats(0.0).tolist() == [1.0] * 8
    shifted = SpanText(np.array([[PAD, 0, 1, 2], [3, 4, 5, PAD]]), TOKEN[:6])
    assert shifted.repeats(1.0).tolist() == [0.5] * 6
    unspanned = SpanText(np.array([[0, 1, 2]]), TOKEN)
    assert unspanned.repeats(1.0).tolist() == [1.0] * 8


def test_novel() -> None:
    text = SpanText(INDEX, TOKEN)
    assert text.novel(frozenset()).tolist() == list(range(8))
    assert text.novel(frozenset({(5, 6, 7)})).tolist() == [6, 7]
    assert len(SpanText(np.array([[0, PAD]]), TOKEN).novel(frozenset())) == 0

import numpy as np
import pytest

from megspeech.evaluation.studies.pos.grammar import Grammar

ALTERNATING = Grammar.of(np.array([0, 1, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 1, 1]), np.arange(6.0), 2)


def test_grammar_of() -> None:
    grammar = Grammar.of(np.array([0, 1, 1]), np.array([0, 0, 1]), np.arange(3.0), 2)
    assert np.allclose(np.exp(grammar.transition), [[1 / 3, 2 / 3], [1 / 2, 1 / 2]])
    assert np.exp(ALTERNATING.start).tolist() == pytest.approx([0.75, 0.25])
    assert np.allclose(np.exp(ALTERNATING.transition), [[1 / 4, 3 / 4], [3 / 4, 1 / 4]])


def test_viterbi() -> None:
    assert ALTERNATING.viterbi(np.zeros((4, 2))).tolist() == [0, 1, 0, 1]
    assert ALTERNATING.viterbi(np.array([[0.0, 5.0], [0.0, 0.0]])).tolist() == [1, 0]


def test_decode() -> None:
    decoded = ALTERNATING.decode(np.zeros((5, 2)), np.array([1, 0, 1, 0, 1]), np.array([2.0, 1.0, 0.0, 0.0, 1.0]))
    assert decoded.tolist() == [0, 1, 0, 0, 1]

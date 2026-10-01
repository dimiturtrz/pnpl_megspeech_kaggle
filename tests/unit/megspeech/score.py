import numpy as np
import pytest

from megspeech.score import BalancedTopK


def test_call() -> None:
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    labels = np.array([0, 2, 2])
    assert BalancedTopK(1)(scores, labels) == pytest.approx(0.5)
    assert BalancedTopK(3)(scores, labels) == pytest.approx(1.0)

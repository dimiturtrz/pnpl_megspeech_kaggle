import numpy as np
import pytest

from megspeech.evaluation.studies.signal.acoustic import Envelopes


def envelopes() -> Envelopes:
    truth = np.array([[0.0, 1.0, 2.0, 3.0], [np.nan] * 4, [3.0, 1.0, 2.0, 0.0]], np.float32)
    predicted = np.array([[0.0, 2.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]])
    return Envelopes(predicted, truth)


def test_present() -> None:
    present = envelopes().present()
    assert present.predicted.shape == (2, 4)
    assert np.isfinite(present.truth).all()


def test_windows() -> None:
    both = envelopes().windows()
    assert both == pytest.approx((1.0 + np.corrcoef([0, 1, 2, 3], [3, 1, 2, 0])[0, 1]) / 2)
    flat = Envelopes(np.ones((1, 4)), np.arange(4, dtype=np.float32)[None])
    assert np.isnan(flat.windows())


def test_pooled() -> None:
    assert envelopes().pooled() == pytest.approx(np.corrcoef([0, 2, 4, 6, 0, 1, 2, 3], [0, 1, 2, 3, 3, 1, 2, 0])[0, 1])

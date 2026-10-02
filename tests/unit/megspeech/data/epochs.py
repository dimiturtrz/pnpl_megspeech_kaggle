import numpy as np

from megspeech.data.epochs import DECIMATE, WINDOW, Window

CHANNELS = 3


def test_window_call() -> None:
    raw = np.ones((2, CHANNELS, WINDOW)) * 1e-12
    out = Window()(raw)
    assert out.shape == (2, CHANNELS, WINDOW // DECIMATE)
    assert out.dtype == np.float32
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-5)

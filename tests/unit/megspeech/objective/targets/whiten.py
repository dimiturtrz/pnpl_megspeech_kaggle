import numpy as np

from megspeech.objective.targets.whiten import Whitening

ROWS = 400
DIM = 8


def runs() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    scale = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])
    first, second = (rng.normal(size=(ROWS, DIM)) * scale + 30.0 for _ in range(2))
    first[3] = np.nan
    return [first.astype(np.float16), second.astype(np.float16)]


def test_fit() -> None:
    """The kept directions are those carrying the variance share, fewer than all of them."""
    fitted = Whitening.fit(runs())
    assert 0 < fitted.basis.shape[1] < DIM
    assert fitted.basis.shape[0] == DIM


def test_apply() -> None:
    """Whitened rows have zero mean and unit variance per direction; an absent row stays absent."""
    fitted = Whitening.fit(runs())
    out = fitted.apply(runs()[0])
    assert np.isnan(out[3]).all()
    finite = out[np.isfinite(out).all(axis=1)].astype(np.float32)
    assert np.abs(finite.mean(0)).max() < 0.3
    assert np.allclose(finite.std(0), 1.0, atol=0.3)

import numpy as np
import torch

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.model.config import Covariance
from megspeech.training.regularise.bgmix import MIN_CELL
from megspeech.training.regularise.noise import SAMPLES, NoiseSample, TotalSample

CPU = torch.device("cpu")
THIN = 2


def split() -> Split:
    labels = np.array([3] * MIN_CELL + [4] * THIN)
    amplitude = np.where(np.arange(len(labels)) % 2, 6.0, 4.0)
    amplitude[MIN_CELL:] = 7.0
    x = (amplitude[:, None, None] * np.array([-1.0, 0.0, 1.0])).repeat(2, axis=1).astype(np.float32)
    return Split.from_arrays(x, labels, Spans.by_gap(np.arange(len(labels)) * 10.0), CPU)


def test_total_sample_windows() -> None:
    sample = torch.ones(4, 2, 3)
    assert TotalSample().windows(split(), sample) is sample


def test_noise_sample_windows() -> None:
    """A rich word's residual is its window less its mean; a thin word's, less the grand mean."""
    train = split()
    residual = NoiseSample().windows(train, torch.zeros(len(train.y), 1, 1))
    assert residual.shape == (len(train.y), 2, 3)
    size = residual[:, 0, 2].abs().sort().values
    assert torch.allclose(size[:MIN_CELL], torch.ones(MIN_CELL), atol=1e-5)
    assert torch.allclose(
        size[MIN_CELL:], torch.full((THIN,), 7.0 - (5.0 * MIN_CELL + 7.0 * THIN) / len(size)), atol=1e-5
    )


def test_samples_cover_every_covariance() -> None:
    assert set(SAMPLES) == set(Covariance)

import pytest
import torch

from megspeech.data.config import DataConfig
from megspeech.training.regularise.stretch import Stretch

PIVOT = 10
SAMPLES = 50


def impulses(n: int) -> torch.Tensor:
    x = torch.zeros(n, 3, SAMPLES)
    x[..., PIVOT] = 1.0
    return x


def test_post_init() -> None:
    with pytest.raises(ValueError, match="stretch"):
        Stretch(1.0)
    with pytest.raises(ValueError, match="stretch"):
        Stretch(-0.1)


def test_of() -> None:
    assert Stretch.of(0.1, DataConfig(delay=0.2)) == Stretch(0.1, -DataConfig(delay=0.2).cut().offset)
    assert Stretch.of(0.1, DataConfig(delay=0.2)).pivot < 0
    assert Stretch.of(0.0, DataConfig()).pivot == 0


def test_call() -> None:
    """Off is the very tensor; on keeps shape and dtype float32, holds the onset sample, and stretches each window by
    its own factor."""
    x = torch.randn(4, 3, SAMPLES, generator=torch.Generator().manual_seed(0))
    assert Stretch(0.0, PIVOT)(x) is x
    torch.manual_seed(0)
    stretched = Stretch(0.2, PIVOT)(impulses(64).half())
    assert stretched.shape == (64, 3, SAMPLES)
    assert stretched.dtype == torch.float32
    assert torch.equal(stretched[..., PIVOT], torch.ones(64, 3))
    ramp = torch.arange(SAMPLES, dtype=torch.float32).expand(64, 1, -1) - PIVOT
    slopes = Stretch(0.2, PIVOT)(ramp)[:, 0, PIVOT + 1]
    assert slopes.min() >= 1 / 1.2 - 1e-5
    assert slopes.max() <= 1 / 0.8 + 1e-5
    assert len(slopes.unique()) > 1


def test_resampled() -> None:
    """Unit factor is the identity; factor 2 reads each later sample at half its latency from the onset, and a factor
    below 1 reads past the window's end as zero."""
    x = torch.randn(2, 3, SAMPLES, generator=torch.Generator().manual_seed(1))
    stretch = Stretch(0.5, PIVOT)
    assert torch.allclose(stretch.resampled(x, torch.ones(2, 1)), x)
    doubled = stretch.resampled(x, torch.full((2, 1), 2.0))
    assert torch.allclose(doubled[..., PIVOT + 8], x[..., PIVOT + 4])
    assert torch.allclose(doubled[..., PIVOT + 7], (x[..., PIVOT + 3] + x[..., PIVOT + 4]) / 2)
    assert torch.allclose(doubled[..., PIVOT - 4], x[..., PIVOT - 2])
    halved = stretch.resampled(x, torch.full((2, 1), 0.5))
    assert torch.allclose(halved[..., PIVOT + 2], x[..., PIVOT + 4])
    assert halved[..., PIVOT + (SAMPLES - PIVOT) // 2 + 1 :].eq(0).all()
    assert halved[..., : PIVOT // 2].eq(0).all()

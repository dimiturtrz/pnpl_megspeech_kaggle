import pytest
import torch

from megspeech.model.words.layers.downsample import Downsample

GENERATOR = torch.Generator().manual_seed(0)


@pytest.mark.parametrize(("samples", "expected"), [(50, 25), (51, 26), (1, 1)])
def test_forward(samples: int, expected: int) -> None:
    layer = Downsample(8, 12)
    x = torch.randn(4, 8, samples, generator=GENERATOR)
    assert layer(x).shape == (4, 12, expected)
    assert layer(x, torch.tensor([True, True, True, False])).shape == (4, 12, expected)

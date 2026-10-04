import torch

from megspeech.data.constants import CHANNELS
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.expert import Crop, Expert

SMALL = ConvConfig(hidden=8, dilations=(1,))
SAMPLES = 20
GENERATOR = torch.Generator().manual_seed(0)


def window() -> torch.Tensor:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR) + 2.0
    x[0, :, 15:] = 0
    return x


def test_crop() -> None:
    cut = Crop(10, 8)(window())
    assert cut.shape == (3, CHANNELS, 8)
    assert torch.equal(cut[0, :, 5:], torch.zeros(CHANNELS, 3))
    assert torch.allclose(cut[0, :, :5].mean(-1), torch.zeros(CHANNELS), atol=1e-5)
    assert torch.allclose(cut[1:].mean(-1), torch.zeros(2, CHANNELS), atol=1e-5)


def test_embed() -> None:
    x = window()
    expert = Expert(Crop(5, 10), SMALL, CHANNELS).eval()
    expected = expert.words.embed(expert.scale(Crop(5, 10)(x)))
    assert expected.shape == (3, SMALL.width)
    assert torch.allclose(expert.embed(x), expected)

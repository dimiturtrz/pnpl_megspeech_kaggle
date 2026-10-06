import torch

from megspeech.model.words.layers.dilated import DilatedBlock
from megspeech.model.words.layers.sensorwise import Sensorwise

CHANNELS = 5
WIDTH = 4
HIDDEN = 8
SAMPLES = 20
GENERATOR = torch.Generator().manual_seed(0)


def make() -> Sensorwise:
    return Sensorwise(CHANNELS, WIDTH, HIDDEN, (DilatedBlock(WIDTH, d, 0.0) for d in (1, 2)))


def test_forward() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert make().forward(x).shape == (3, HIDDEN, SAMPLES)


def test_features() -> None:
    net = make().eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    perm = torch.randperm(CHANNELS, generator=GENERATOR)
    assert net.features(x).shape == (3, CHANNELS, WIDTH, SAMPLES)
    assert torch.allclose(net.features(x[:, perm]), net.features(x)[:, perm], atol=1e-6)
    net = net.train()
    x = torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)
    valid = torch.tensor([True, True, False, False])
    padded = x.clone()
    padded[2:] = 100.0
    assert torch.allclose(net.features(padded, valid)[:2], net.features(x[:2], valid[:2])[:2], atol=1e-5)

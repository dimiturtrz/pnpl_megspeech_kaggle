import pytest
import torch

from megspeech.model.words.layers.dilated import DilatedBlock
from megspeech.model.words.layers.sensorwise import Grouping, Sensorwise

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


def make_site() -> Sensorwise:
    return Sensorwise(6, WIDTH, HIDDEN, (DilatedBlock(WIDTH, d, 0.0) for d in (1, 2)), Grouping(3))


def test_site_forward() -> None:
    x = torch.randn(3, 6, SAMPLES, generator=GENERATOR)
    assert make_site().forward(x).shape == (3, HIDDEN, SAMPLES)


def test_site_features() -> None:
    net = make_site().eval()
    x = torch.randn(3, 6, SAMPLES, generator=GENERATOR)
    zeroed = x.clone()
    zeroed[:, 3:6] = 0.0
    base, changed = net.features(x), net.features(zeroed)
    assert base.shape == (3, 2, WIDTH, SAMPLES)
    assert torch.allclose(base[:, 0], changed[:, 0], atol=1e-6)
    assert not torch.allclose(base[:, 1], changed[:, 1], atol=1e-6)


def test_site_indivisible() -> None:
    with pytest.raises(ValueError, match="sites of 3"):
        Sensorwise(5, WIDTH, HIDDEN, (), Grouping(3))


def test_site_params() -> None:
    count = sum(p.numel() for p in Sensorwise(306, 8, 256, ()).parameters())
    assert count == 8 * 2 + 306 * 8 + 306 * 8 * 256 + 256
    assert sum(p.numel() for p in Sensorwise(306, 8, 256, (), Grouping(3)).parameters()) < count


def test_out() -> None:
    net = Sensorwise(306, 16, 256, (), Grouping(3, 2))
    assert net.fuse.in_channels == 204
    assert net.sensor.shape == (102, 2, 1)
    assert net.forward(torch.randn(2, 306, SAMPLES, generator=GENERATOR)).shape == (2, 256, SAMPLES)
    assert isinstance(Sensorwise(306, 16, 256, ()).reduce, torch.nn.Identity)

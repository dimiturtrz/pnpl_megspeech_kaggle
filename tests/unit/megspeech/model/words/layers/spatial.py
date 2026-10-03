import torch

from megspeech.data.constants import CHANNELS
from megspeech.model.words.layers.spatial import FREQUENCIES, RADIUS, SpatialAttention

GENERATOR = torch.Generator().manual_seed(0)


def test_positions() -> None:
    xy = SpatialAttention.positions()
    assert xy.shape == (CHANNELS, 2)
    assert xy.amin(0).tolist() == [0.0, 0.0]
    assert xy.amax(0).tolist() == [1.0, 1.0]
    assert len(xy.unique(dim=0)) == CHANNELS


def test_features() -> None:
    xy = torch.tensor([[0.0, 0.0], [0.5, 0.25]])
    features = SpatialAttention.features(xy)
    assert features.shape == (2, 2 * FREQUENCIES**2)
    assert torch.equal(features[0], torch.cat([torch.ones(FREQUENCIES**2), torch.zeros(FREQUENCIES**2)]))
    assert torch.allclose(features[1, FREQUENCIES], torch.tensor(-1.0))


def test_kept() -> None:
    layer = SpatialAttention(4)
    x = torch.randn(64, CHANNELS, 5, generator=GENERATOR)
    assert layer.eval().kept(x).eq(1).all()
    kept = layer.train().kept(x)
    assert set(kept.unique().tolist()) == {0.0, 1.0}
    assert (kept.sum(1) > CHANNELS * (1 - torch.pi * RADIUS**2) / 2).all()


def test_forward() -> None:
    layer = SpatialAttention(4).eval()
    x = torch.randn(3, CHANNELS, 5, generator=GENERATOR)
    merged = layer.forward(x)
    assert merged.shape == (3, 4, 5)
    weights = (layer.scores @ layer.fourier.T).softmax(1)
    assert torch.allclose(merged, torch.einsum("oc,bct->bot", weights, x), atol=1e-5)
    constant = torch.full((2, CHANNELS, 5), 3.0)
    assert torch.allclose(layer.train().forward(constant), torch.full((2, 4, 5), 3.0), atol=1e-5)


def test_forward_reads_zero_where_every_weighted_sensor_is_dropped() -> None:
    layer = SpatialAttention(1).train()
    with torch.no_grad():
        layer.scores.copy_(1e4 * layer.fourier[:1])
    merged = layer.forward(torch.ones(256, CHANNELS, 5))
    assert torch.isfinite(merged).all()
    assert merged.eq(0).any()

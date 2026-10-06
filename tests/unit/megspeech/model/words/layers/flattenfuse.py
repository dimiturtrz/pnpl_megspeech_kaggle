import torch

from megspeech.model.words.layers.flattenfuse import FlattenFuse

SITES = 6
DIM = 4
HIDDEN = 8
SAMPLES = 10
GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    fuse = FlattenFuse(SITES, DIM, HIDDEN, 0.5)
    h = torch.randn(3, SITES, DIM, SAMPLES, generator=GENERATOR)
    plain = torch.nn.functional.conv1d(h.flatten(1, 2), fuse.weight, fuse.bias)
    assert fuse.forward(h).shape == (3, HIDDEN, SAMPLES)
    assert torch.equal(fuse.forward(h), plain)
    keep = torch.tensor([[True, False] * 3] * 3)
    kept = h * keep[:, :, None, None] * 2.0
    expected = torch.nn.functional.conv1d(kept.flatten(1, 2), fuse.weight, fuse.bias)
    assert torch.allclose(fuse.forward(h, keep), expected, atol=1e-6)
    probe = h.clone()
    probe[:, 1] += 9.0
    assert torch.allclose(fuse.forward(probe, keep), fuse.forward(h, keep), atol=1e-5)

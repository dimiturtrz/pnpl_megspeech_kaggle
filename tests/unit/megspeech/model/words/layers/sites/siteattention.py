import torch

from megspeech.model.words.layers.sites.siteattention import SiteAttention

SAMPLES = 20
GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    layer = SiteAttention(4, 1, 2, 0.0).eval()
    h = torch.randn(2, 5, 4, SAMPLES, generator=GENERATOR)
    out = layer.forward(h)
    assert out.shape == h.shape
    perm = torch.randperm(5, generator=GENERATOR)
    assert torch.allclose(layer.forward(h[:, perm]), out[:, perm], atol=1e-5)
    assert not torch.allclose(out[:, 0], h[:, 0])
    keep = torch.tensor([[True, True, True, False, False]] * 2)
    probe = h.clone()
    probe[:, 3:] += 5.0
    assert torch.allclose(layer.forward(probe, keep)[:, :3], layer.forward(h, keep)[:, :3], atol=1e-5)

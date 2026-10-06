import torch

from megspeech.model.words.layers.setpool import SetPool

SITES = 6
DIM = 4
HIDDEN = 8
SAMPLES = 10
GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    pool = SetPool(DIM, 3, 2, HIDDEN).eval()
    h = torch.randn(3, SITES, DIM, SAMPLES, generator=GENERATOR)
    assert pool.forward(h).shape == (3, HIDDEN, SAMPLES)
    perm = torch.randperm(SITES, generator=GENERATOR)
    assert torch.allclose(pool.forward(h[:, perm]), pool.forward(h), atol=1e-5)
    keep = torch.tensor([[True, True, False, True, False, True]] * 3)
    probe = h.clone()
    probe[:, 2] += 9.0
    probe[:, 4] -= 9.0
    assert torch.allclose(pool.forward(probe, keep), pool.forward(h, keep), atol=1e-5)
    assert not torch.allclose(pool.forward(probe), pool.forward(h), atol=1e-3)
    assert not torch.allclose(pool.forward(h, keep), pool.forward(h), atol=1e-6)
    assert pool.forward(torch.randn(2, 102, DIM, SAMPLES)).shape == (2, HIDDEN, SAMPLES)

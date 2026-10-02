import torch

from megspeech.model.conformer import Conformer

GENERATOR = torch.Generator().manual_seed(0)


def test_feedforward() -> None:
    feedforward = Conformer.feedforward(8, 0.0)
    assert feedforward(torch.randn(2, 5, 8, generator=GENERATOR)).shape == (2, 5, 8)


def test_forward() -> None:
    """An impulse mid-window reaches both window edges: attention mixes every sample with every other."""
    conformer = Conformer(8, 0.0, heads=2, kernel=3).eval()
    x = torch.zeros(2, 8, 30)
    x[..., 15] = 1.0
    with torch.no_grad():
        mixed = conformer.forward(x)
        baseline = conformer.forward(torch.zeros_like(x))
    assert mixed.shape == x.shape
    assert (mixed - baseline)[..., 0].abs().sum() > 0
    assert (mixed - baseline)[..., -1].abs().sum() > 0


def test_forward_valid() -> None:
    conformer = Conformer(8, 0.0, heads=2).train()
    x = torch.randn(3, 8, 30, generator=GENERATOR)
    valid = torch.tensor([True, True, False])
    assert torch.allclose(conformer.forward(x, valid)[:2], conformer.forward(x[:2]), atol=1e-5)

import torch

from megspeech.model.scaler import CLAMP, Scaler


def test_forward() -> None:
    train = torch.randn(100, 2, 10, generator=torch.Generator().manual_seed(0))
    out = Scaler(train).forward(torch.full((1, 2, 10), 1e6))
    assert torch.all(out == CLAMP)


def test_forward_flat_channel() -> None:
    train = torch.randn(100, 2, 10, generator=torch.Generator().manual_seed(0))
    train[:, 1] = 3.0
    scaler = Scaler(train)
    x = torch.randn(4, 2, 10, generator=torch.Generator().manual_seed(1))
    out = scaler.forward(x)
    assert torch.isfinite(out).all()
    assert torch.equal(out[:, 0], ((x[:, 0] - scaler.median[0]) / scaler.spread[0]).clamp(-CLAMP, CLAMP))
    assert torch.equal(scaler.forward(torch.full((1, 2, 10), 3.0))[:, 1], torch.zeros(1, 10))

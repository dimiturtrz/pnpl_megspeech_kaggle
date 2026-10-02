import torch

from megspeech.model.scaler import CLAMP, Scaler


def test_forward() -> None:
    train = torch.randn(100, 2, 10, generator=torch.Generator().manual_seed(0))
    out = Scaler(train).forward(torch.full((1, 2, 10), 1e6))
    assert torch.all(out == CLAMP)

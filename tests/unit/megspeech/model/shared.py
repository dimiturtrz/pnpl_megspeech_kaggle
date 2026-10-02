import torch

from megspeech.model.shared import Shared

GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    x = torch.randn(2, 4, 3, generator=GENERATOR)
    assert torch.equal(Shared().forward(x, torch.tensor([0, 5])), x)

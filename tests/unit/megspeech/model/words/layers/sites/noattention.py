import torch

from megspeech.model.words.layers.sites.noattention import NoAttention


def test_forward() -> None:
    h = torch.randn(2, 5, 4, 7, generator=torch.Generator().manual_seed(0))
    keep = torch.tensor([[True, True, True, False, False]] * 2)
    assert torch.equal(NoAttention().forward(h, keep), h)
    assert torch.equal(NoAttention().forward(h), h)

import torch

from megspeech.block import Block

GENERATOR = torch.Generator().manual_seed(0)


def test_block_forward() -> None:
    block = Block(4, 2, 0.0).eval()
    x = torch.randn(2, 4, 10, generator=GENERATOR)
    assert block.forward(x).shape == x.shape


def test_block_forward_valid() -> None:
    block = Block(4, 2, 0.0).train()
    x = torch.randn(3, 4, 10, generator=GENERATOR)
    valid = torch.tensor([True, True, False])
    assert torch.allclose(block.forward(x, valid)[:2], block.forward(x[:2]), atol=1e-5)

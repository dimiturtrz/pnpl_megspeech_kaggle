import torch

from megspeech.block import Block

GENERATOR = torch.Generator().manual_seed(0)


def test_block_forward() -> None:
    block = Block(4, 2, 0.0).eval()
    x = torch.randn(2, 4, 10, generator=GENERATOR)
    assert block.forward(x).shape == x.shape

import torch

from megspeech.model.words.layers.dilated import DilatedBlock

GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    block = DilatedBlock(4, 2, 0.0).eval()
    x = torch.randn(2, 4, 10, generator=GENERATOR)
    assert block.forward(x).shape == x.shape


def test_forward_valid() -> None:
    block = DilatedBlock(4, 2, 0.0).train()
    x = torch.randn(3, 4, 10, generator=GENERATOR)
    valid = torch.tensor([True, True, False])
    assert torch.allclose(block.forward(x, valid)[:2], block.forward(x[:2]), atol=1e-5)


def test_block_keeps_checkpoint_keys() -> None:
    block = DilatedBlock(4, 2, 0.0)
    assert block.body[0].kernel_size == (3,)
    assert {"body.0.weight", "body.0.bias"} <= set(block.state_dict())

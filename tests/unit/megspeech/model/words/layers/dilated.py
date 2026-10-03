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


def test_conv() -> None:
    conv = DilatedBlock.conv(4, 2, 5)
    x = torch.randn(2, 4, 10, generator=GENERATOR)
    assert conv.kernel_size == (5,)
    assert conv(x).shape == x.shape


def test_block_default_keeps_checkpoint_keys() -> None:
    keys = set(DilatedBlock(4, 2, 0.0).state_dict())
    assert "body.0.weight" in keys
    assert not any(key.startswith("parallel.") for key in keys)


def test_forward_parallel_kernels() -> None:
    block = DilatedBlock(4, 2, 0.0, (3, 5, 7)).eval()
    x = torch.randn(2, 4, 10, generator=GENERATOR)
    assert block.body[0].kernel_size == (3,)
    assert [conv.kernel_size for conv in block.parallel] == [(5,), (7,)]
    assert block.forward(x).shape == x.shape

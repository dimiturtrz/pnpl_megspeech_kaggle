import pytest
import torch
from torch import nn

from megspeech.model.context.droppath import DropPath, DropPathLayer

GENERATOR = torch.Generator().manual_seed(0)
PADDING = torch.tensor([[False, False, True], [False, True, True]])


def test_forward() -> None:
    """Rate 0 is the plain layer's output bit for bit; evaluation ignores the rate; a rate of 1 - eps keeps the
    skip path only."""
    layer = DropPathLayer(8, 2, 16, 0.0)
    plain = nn.TransformerEncoderLayer(8, 2, 16, 0.0, batch_first=True, norm_first=True)
    plain.load_state_dict(layer.state_dict(), strict=False)
    h = torch.randn(2, 3, 8, generator=GENERATOR)
    expected = plain.train()(h, src_key_padding_mask=PADDING)
    assert torch.equal(layer.train()(h, src_key_padding_mask=PADDING), expected)
    layer.path.rate = 0.5
    assert torch.equal(layer.eval()(h, src_key_padding_mask=PADDING), plain.eval()(h, src_key_padding_mask=PADDING))
    layer.path.rate = 1.0 - 1e-9
    assert torch.allclose(layer.train()(h, src_key_padding_mask=PADDING), h)


def test_path() -> None:
    path = DropPath(0.5)
    branch = torch.ones(512, 3, 4)
    kept = path.train()(branch)
    assert set(kept.unique().tolist()) == {0.0, 2.0}
    assert kept.flatten(1).std(dim=1).max() == 0
    assert torch.equal(path.eval()(branch), branch)
    assert torch.equal(DropPath(0.0).train()(branch), branch)


def test_ramp() -> None:
    paths = [DropPath() for _ in range(4)]
    DropPath.ramp(paths, 0.2)
    assert [p.rate for p in paths] == pytest.approx([0.05, 0.1, 0.15, 0.2])


def test_residual() -> None:
    base, out = torch.zeros(256, 2, 4), torch.ones(256, 2, 4)
    path = DropPath(0.5)
    assert set(path.train().residual(base, out).unique().tolist()) == {0.0, 2.0}
    assert path.eval().residual(base, out) is out
    assert DropPath(0.0).train().residual(base, out) is out

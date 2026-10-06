import pytest
import torch

from megspeech.model.context.droppath import DropPath


def test_forward() -> None:
    path = DropPath(0.5)
    branch = torch.ones(512, 3, 4)
    kept = path.train().forward(branch)
    assert set(kept.unique().tolist()) == {0.0, 2.0}
    assert kept.flatten(1).std(dim=1).max() == 0
    assert torch.equal(path.eval().forward(branch), branch)
    assert torch.equal(DropPath(0.0).train().forward(branch), branch)


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

from pathlib import Path

import torch

from core.paths import DataRoot
from megspeech.training.regularise.regional import Regional

POSITIONS = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]])


def test_of(tmp_path: Path) -> None:
    regional = Regional.of(0.0, DataRoot(tmp_path), torch.device("cpu"))
    assert not regional.near.numel()


def test_off() -> None:
    x = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(Regional.off()(x), x)


def test_within() -> None:
    near = Regional.within(POSITIONS, 0.2).near
    expected = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(near, expected)


def test_call() -> None:
    regional = Regional.within(POSITIONS, 0.2)
    torch.manual_seed(0)
    x = torch.ones(64, 4, 3)
    dropped = regional(x)
    alive = dropped[..., 0] != 0
    assert torch.equal(dropped, x * alive[..., None])
    patches = regional.near.tolist()
    assert all(row in patches for row in (~alive).tolist())
    assert len({tuple(row) for row in (~alive).tolist()}) == len({tuple(row) for row in patches})

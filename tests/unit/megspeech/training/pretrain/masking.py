import pytest
import torch

from megspeech.training.pretrain.masking import Channels, Masking, Region

RATE = 50.0
LINE = torch.stack([torch.arange(10.0), torch.zeros(10), torch.zeros(10)], dim=1)


def test_patches() -> None:
    assert Masking(patch=0.1).patches(RATE) == 5
    assert Masking(patch=0.001).patches(RATE) == 1


def test_draw() -> None:
    hidden = Masking(ratio=0.4, patch=0.1).draw(64, 50, RATE, torch.Generator().manual_seed(0))
    assert hidden.shape == (64, 50)
    assert hidden.dtype == torch.bool
    assert (hidden.sum(dim=1) == 20).all()
    assert hidden.reshape(64, 10, 5).all(dim=2).sum(dim=1).eq(4).all()
    assert hidden.any(dim=0).all()


def test_draw_ragged() -> None:
    hidden = Masking(ratio=0.5, patch=0.12).draw(4, 50, RATE, torch.Generator().manual_seed(1))
    assert hidden.shape == (4, 50)
    assert hidden.any(dim=1).all()


def test_apply() -> None:
    x = torch.ones(2, 3, 10)
    hidden = torch.zeros(2, 10, dtype=torch.bool)
    hidden[:, 4:6] = True
    out = Masking(dropout=0.0).apply(x, hidden, torch.full((3, 1), -2.0), torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert (out[:, :, 4:6] == -2).all()
    assert (out[:, :, :4] == 1).all()


def test_apply_dropout() -> None:
    out = Masking(dropout=0.5).apply(
        torch.ones(64, 20, 4),
        torch.zeros(64, 4, dtype=torch.bool),
        torch.zeros(20, 1),
        torch.Generator().manual_seed(0),
    )
    zeroed = out.eq(0).all(dim=2)
    assert 0.35 < zeroed.float().mean().item() < 0.65
    assert zeroed.eq(out.eq(0).any(dim=2)).all()


def test_validates() -> None:
    with pytest.raises(ValueError, match="ratio"):
        Masking(ratio=1.0)
    with pytest.raises(ValueError, match="dropout"):
        Masking(dropout=1.0)
    with pytest.raises(ValueError, match="region"):
        Masking(region=0)


def test_apply_region() -> None:
    out = Masking(channels=Channels.REGION, dropout=0.9).apply(
        torch.ones(8, 20, 4), torch.zeros(8, 4, dtype=torch.bool), torch.zeros(20, 1), torch.Generator().manual_seed(0)
    )
    assert out.eq(1).all()


def test_silence() -> None:
    x = torch.ones(16, 10, 6)
    region = Region.of(LINE, 2)
    assert Masking(channels=Channels.DROPOUT).silence(x, region, torch.Generator().manual_seed(0)) is x
    out = Masking(channels=Channels.REGION, region=2).silence(x, region, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    zeroed = out.eq(0).all(dim=2)
    assert zeroed.eq(out.eq(0).any(dim=2)).all()
    assert (zeroed.sum(dim=1) == 3).all()
    assert (x == 1).all()


def test_of() -> None:
    nearest = Region.of(LINE, 2).nearest
    assert nearest.shape == (10, 3)
    assert set(nearest[5].tolist()) == {4, 5, 6}
    assert set(nearest[0].tolist()) == {0, 1, 2}
    assert set(nearest[9].tolist()) == {7, 8, 9}
    assert all(i in nearest[i].tolist() for i in range(10))
    twins = Region.of(torch.zeros(4, 3), 1).nearest
    assert all(i in twins[i].tolist() for i in range(4))
    assert Region.of(LINE, 99).nearest.shape == (10, 10)


def test_silent() -> None:
    region = Region.of(LINE, 2)
    silent = region.silent(64, torch.Generator().manual_seed(0))
    assert silent.shape == (64, 10)
    assert silent.dtype == torch.bool
    assert (silent.sum(dim=1) == 3).all()
    for row in silent:
        where = row.nonzero().flatten()
        assert where.max() - where.min() == 2
    assert silent.any(dim=0).all()
    assert torch.equal(silent, region.silent(64, torch.Generator().manual_seed(0)))

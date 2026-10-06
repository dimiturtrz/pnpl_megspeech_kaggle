import pytest
import torch

from megspeech.training.pretrain.masking import Masking

RATE = 50.0


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

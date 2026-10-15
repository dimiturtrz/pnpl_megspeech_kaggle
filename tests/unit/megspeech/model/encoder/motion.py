from torch import nn

from megspeech.model.encoder.motion import Frozen, Moving


def stage() -> tuple[nn.Module, ...]:
    return (nn.Sequential(nn.Linear(2, 2), nn.Dropout()), nn.BatchNorm1d(2))


def test_hold() -> None:
    moving = tuple(module.train() for module in stage())
    Moving().hold(moving)
    assert all(module.training for module in moving)
    assert all(p.requires_grad for module in moving for p in module.parameters())
    frozen = tuple(module.train() for module in stage())
    Frozen().hold(frozen)
    assert not any(m.training for module in frozen for m in module.modules())
    assert not any(p.requires_grad for module in frozen for p in module.parameters())

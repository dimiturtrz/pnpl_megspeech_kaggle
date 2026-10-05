import pytest
import torch
from torch import nn

from megspeech.objective.auxiliary.adversarial import Adversary
from megspeech.training.ramp import Ramp


def ramp() -> Ramp:
    return Ramp.of(nn.Sequential(Adversary(4), nn.Linear(2, 2)), 2.0, 4)


def test_of() -> None:
    assert len(ramp().reversals) == 1
    assert Ramp.of(nn.Linear(2, 2), 1.0, 0).over == 1


def test_strength() -> None:
    hook = ramp()
    assert hook.strength() == 0.0
    hook.taken = 2
    assert hook.strength() == pytest.approx(1.0)
    hook.taken = 99
    assert hook.strength() == 2.0


def test_step() -> None:
    hook = ramp()
    for _ in range(2):
        hook.step(nn.Identity(), 0.0)
    assert hook.reversals[0].strength.item() == pytest.approx(1.0)
    assert torch.isfinite(hook.reversals[0].strength)

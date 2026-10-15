import pytest
import torch
from torch import nn

from megspeech.objective.auxiliary.adversary.adversarial import Adversary, AdversaryTarget
from megspeech.training.adversary import AdversaryConfig, Ramp


def ramp() -> Ramp:
    return Ramp.of(nn.Sequential(Adversary(4), nn.Linear(2, 2)), 2.0, 4)


def test_of() -> None:
    assert len(ramp().reversals) == 1
    assert Ramp.of(nn.Linear(2, 2), 1.0, 0).over == 1


def test_ramp() -> None:
    made = AdversaryConfig(adversary=0.5, adversary_ramp=2.0).ramp(nn.Sequential(Adversary(4)), total=30, epochs=4)
    assert (made.target, made.over, len(made.reversals)) == (0.5, 15, 1)
    assert AdversaryConfig().ramp(nn.Linear(2, 2), total=10, epochs=40).over == 1


def test_adversary_target() -> None:
    """Timing unless a run asks for another target, so every config saved before the run target trains as it did."""
    assert AdversaryConfig().adversary_target is AdversaryTarget.TIMING
    assert AdversaryConfig(adversary_target=AdversaryTarget.RUN).adversary_target is AdversaryTarget.RUN


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

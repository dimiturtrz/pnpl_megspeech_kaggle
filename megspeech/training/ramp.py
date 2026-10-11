"""The gradient reversal's strength, raised linearly from 0 to its target over the first steps of a run."""

from dataclasses import dataclass, field
from typing import ClassVar, override

import torch
from torch import nn

from megspeech.objective.auxiliary.reversal import Reversal
from megspeech.training.hook import StepHook


@dataclass
class Ramp(StepHook):
    """Sets every `Reversal` of a model to `target * min(1, steps taken / over)` after each optimiser step: the
    adversary first learns to read timing, and only then does the encoder get its reversed gradient."""

    reversals: list[Reversal]
    target: float
    over: int
    taken: int = field(default=0, init=False)

    @classmethod
    def of(cls, model: nn.Module, target: float, over: int) -> "Ramp":
        return cls([m for m in model.modules() if isinstance(m, Reversal)], target, max(over, 1))

    def strength(self) -> float:
        """The strength the next forward pass reverses by."""
        return self.target * min(1.0, self.taken / self.over)

    @torch.no_grad()
    @override
    def step(self, model: nn.Module, lr: float) -> None:
        self.taken += 1
        for reversal in self.reversals:
            reversal.strength.fill_(self.strength())


@dataclass(frozen=True)
class AdversaryConfig:
    """The timing adversary's reversal; the `optim.adversarial` group, its leaves also under their flat
    `--optim.<name>` flags."""

    FLAT: ClassVar[bool] = True

    adversary: float = 0.0
    """Strength of the timing adversary's gradient reversal (`megspeech.objective.auxiliary.adversarial`): a head
    regresses each word's log duration and sentence position from the `tap`, and the encoder receives its gradient
    times -`adversary`, so the MEG embedding is trained to hold none of the timing the onsets already give; 0 trains
    no adversary (megspeech-7gf.98)."""
    adversary_ramp: float = 2.0
    """Epochs over which the reversal's strength rises linearly from 0 to `adversary`: the adversary must be able to
    read timing before reversing its gradient means anything."""

    def ramp(self, model: nn.Module, total: int, epochs: int) -> Ramp:
        """The ramp raising `model`'s reversals to `adversary` over `adversary_ramp` of a run's `epochs`, `total`
        steps in all."""
        return Ramp.of(model, self.adversary, round(total * self.adversary_ramp / epochs))

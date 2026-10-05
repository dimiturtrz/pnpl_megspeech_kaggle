"""The gradient reversal's strength, raised linearly from 0 to its target over the first steps of a run."""

from dataclasses import dataclass, field
from typing import override

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

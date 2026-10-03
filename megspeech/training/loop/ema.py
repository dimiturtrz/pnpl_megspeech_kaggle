"""An exponential moving average of a model's parameters, scored and kept in place of the raw weights."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn
from torch.optim.swa_utils import get_ema_multi_avg_fn

from megspeech.training.hook import StepHook


@dataclass(frozen=True)
class Ema(StepHook):
    """Shadow parameters moved toward the model's by `1 - decay` each step; decay 0 tracks the model exactly."""

    decay: float
    shadow: list[Tensor]

    @classmethod
    def of(cls, model: nn.Module, decay: float) -> "Ema":
        return cls(decay, [p.detach().clone() for p in model.parameters()])

    @torch.no_grad()
    @override
    def step(self, model: nn.Module, lr: float) -> None:
        """Move the average toward `model` after an optimiser step at `lr`, which the average does not read."""
        get_ema_multi_avg_fn(self.decay)(self.shadow, [p.detach() for p in model.parameters()], None)

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[nn.Module]:
        """`model` holding the average for the block, its own weights restored after; copied in place so the
        tensors a compiled graph captured stay the ones it reads."""
        parameters = [p.detach() for p in model.parameters()]
        own = [p.clone() for p in parameters]
        self.copy(parameters, self.shadow)
        try:
            yield model
        finally:
            self.copy(parameters, own)

    @staticmethod
    @torch.no_grad()
    def copy(into: list[Tensor], values: list[Tensor]) -> None:
        for target, value in zip(into, values, strict=True):
            target.copy_(value)

"""What advances after every optimiser step: the anchor's pull, the learning-rate schedule, the weight average."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import torch
from torch import nn


class StepHook(ABC):
    @abstractmethod
    def step(self, model: nn.Module, lr: float) -> None:
        """Advance after an optimiser step of `model` taken at learning rate `lr`."""


@dataclass(frozen=True)
class Scheduled(StepHook):
    """The learning-rate schedule, stepped after the optimiser."""

    schedule: torch.optim.lr_scheduler.LRScheduler

    def lr(self) -> float:
        """The rate the last optimiser step took, the first parameter group's."""
        return float(self.schedule.get_last_lr()[0])

    @override
    def step(self, model: nn.Module, lr: float) -> None:
        self.schedule.step()

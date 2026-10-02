"""Decoupled decay of the subject matrices toward the Deep subject's instead of toward 0.

AdamW pulls every weight toward 0, which for an identity-initialised sensor map is no prior at all. The anchor pulls
each subject's matrix toward the Deep one's, the matrix a zero-shot subject is read through, so what a subject keeps
of its own is what its words pay for — the train-time form of the eval-time shrink (`SubjectLayer.shrink`).
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from megspeech.model.subjects import SubjectLayer, Unseen


@dataclass(frozen=True)
class Anchor:
    """Each step pulls every subject's matrix toward the Deep one's by `strength` times the step's learning rate,
    as AdamW's decoupled decay pulls toward 0; 0 never pulls."""

    strength: float = 0.0

    @classmethod
    def of(cls, keep: float, lr: float, steps: int) -> "Anchor":
        """The strength under which a deviation no gradient restores keeps `keep` of itself after `steps` steps of
        the run's one-cycle schedule peaking at `lr`: the product of `1 - strength * lr_t` is `exp(-strength * sum
        lr_t)`."""
        probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=lr)
        schedule = torch.optim.lr_scheduler.OneCycleLR(probe, lr, total_steps=steps)
        total = 0.0
        for _ in range(steps):
            total += float(schedule.get_last_lr()[0])
            probe.step()
            schedule.step()
        return cls(math.log(1 / keep) / total)

    def groups(self, model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
        """`model`'s parameters as optimiser groups: the subject matrices decay toward 0 only when not anchored."""
        subject = [m.weight for m in model.modules() if isinstance(m, SubjectLayer)]
        ids = {id(p) for p in subject}
        rest = [p for p in model.parameters() if id(p) not in ids]
        return [{"params": rest}, {"params": subject, "weight_decay": weight_decay * (self.strength == 0)}]

    def pull(self, model: nn.Module, lr: float) -> None:
        """One step's pull of every subject matrix in `model` toward the Deep one's."""
        for layer in (m for m in model.modules() if isinstance(m, SubjectLayer)):
            layer.shrink(1 - self.strength * lr, Unseen.DEEP)

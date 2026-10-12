"""Decoupled decay of the encoder toward the weights it started from instead of toward 0 (L2-SP, Li et al. 2018).

An adapt starts from a stem trained on every subject's words and fine-tunes it on one session per subject; under
AdamW's pull toward 0 nothing holds what the stem learnt, so a longer adapt memorises the adapt set. The pull makes
every step of drift away from the stem a step the adapt words have to pay for.
"""

from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from megspeech.model.input.subjects import SubjectLayer
from megspeech.training.hook import StepHook
from megspeech.training.regularise.anchor import Anchor


@dataclass(frozen=True)
class Pull(StepHook):
    """Each step pulls every one of `params` toward its `start` by `strength` times the step's learning rate; 0 never
    pulls."""

    params: tuple[Tensor, ...] = ()
    start: tuple[Tensor, ...] = ()
    strength: float = 0.0

    @classmethod
    def of(cls, keep: float, lr: float, steps: int, model: nn.Module, params: tuple[Tensor, ...]) -> "Pull":
        """The pull of `params` toward their current values under which a drift no gradient restores keeps `keep` of
        itself after `steps` steps of the one-cycle schedule peaking at `lr`; `model`'s subject matrices are left to
        the anchor."""
        strength = Anchor.of(keep, lr, steps).strength
        subject = {id(m.weight) for m in model.modules() if isinstance(m, SubjectLayer)}
        pulled = tuple(p for p in params if id(p) not in subject) if strength else ()
        return cls(pulled, tuple(p.detach().clone() for p in pulled), strength)

    @override
    def step(self, model: nn.Module, lr: float) -> None:
        if self.params:
            with torch.no_grad():
                torch._foreach_lerp_(list(self.params), list(self.start), self.strength * lr)

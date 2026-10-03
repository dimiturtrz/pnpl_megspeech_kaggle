"""A per-subject linear map of the sensors, so one shared encoder reads every subject's head in its own space."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import override

import torch
from jaxtyping import Float, Int
from pnpl.competition.holdout import N_SUBJECTS
from torch import Tensor, nn

from megspeech.data.raw.runs import DEEP


class Shared(nn.Module):
    """Every subject read through the same sensors, unchanged: the single-subject model, with no parameters;
    `SubjectLayer` gives each subject its own map."""

    @override
    def forward(self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"]) -> Float[Tensor, "r c t"]:
        return x


class Unseen(StrEnum):
    """The sensor map a subject the layer has no matrix for is read through — a zero-shot holdout subject."""

    IDENTITY = "identity"
    """The sensors unmapped, as every subject's matrix started."""
    DEEP = "deep"
    """The Deep subject's matrix, the one trained on the most words."""
    MEAN = "mean"
    """The mean of every subject's matrix."""

    def matrix(self, weight: Float[Tensor, "s c c"]) -> Float[Tensor, "c c"]:
        return FALLBACKS[self](weight)


FALLBACKS: dict[Unseen, Callable[[Tensor], Tensor]] = {
    Unseen.IDENTITY: lambda weight: torch.eye(weight.shape[-1], dtype=weight.dtype, device=weight.device),
    Unseen.DEEP: lambda weight: weight[DEEP],
    Unseen.MEAN: lambda weight: weight.mean(dim=0),
}


@dataclass(frozen=True)
class Fallback:
    """How a loaded subject layer reads subjects: `unseen` for those it has no matrix for, and every matrix keeping
    `keep` of its own deviation from that fallback."""

    unseen: Unseen = Unseen.IDENTITY
    keep: float = 1.0


AS_TRAINED = Fallback()
"""Unseen subjects read through the identity, every subject's matrix as trained."""


class Reading(StrEnum):
    """Which subject a run's words are presented to the model as; every word is still read from its own MEG."""

    OWN = "own"
    UNSEEN = "unseen"
    """A subject past every trained one, as a zero-shot holdout subject is."""

    def subjects(self, subject: Int[Tensor, " n"]) -> Int[Tensor, " n"]:
        return READINGS[self](subject)


READINGS: dict[Reading, Callable[[Tensor], Tensor]] = {
    Reading.OWN: lambda subject: subject,
    Reading.UNSEEN: lambda subject: torch.full_like(subject, N_SUBJECTS),
}


class SubjectLayer(Shared):
    """Each subject's own `channels x channels` matrix over the sensors, identity at initialisation.

    Starting at identity keeps a subject-0 model's behaviour when the layer is first switched on: only the
    gradient from a subject's own words moves its matrix away. A subject past the last matrix is read through
    `unseen`'s.
    """

    def __init__(self, subjects: int, channels: int, unseen: Unseen = Unseen.IDENTITY) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(channels).expand(subjects, channels, channels).clone())
        self.unseen = unseen

    @classmethod
    def of(cls, subjects: int, channels: int, unseen: Unseen = Unseen.IDENTITY) -> nn.Module:
        """A layer for `subjects` subjects; none (`Shared`) at 0, which keeps single-subject checkpoints' keys."""
        return cls(subjects, channels, unseen) if subjects else Shared()

    def shrink(self, keep: float, toward: Unseen) -> None:
        """Pull every subject's matrix toward `toward`'s, keeping `keep` of its own deviation from it."""
        with torch.no_grad():
            target = toward.matrix(self.weight).clone()
            self.weight.lerp_(target.expand_as(self.weight), 1 - keep)

    @override
    def forward(self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"]) -> Float[Tensor, "r c t"]:
        known = len(self.weight)
        table = torch.cat([self.weight, self.unseen.matrix(self.weight)[None]])
        return torch.bmm(table[subject.clamp(max=known)], x)

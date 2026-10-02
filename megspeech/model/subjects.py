"""A per-subject linear map of the sensors, so one shared encoder reads every subject's head in its own space."""

from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.shared import Shared


class SubjectLayer(nn.Module):
    """Each subject's own `channels x channels` matrix over the sensors, identity at initialisation.

    Starting at identity keeps a subject-0 model's behaviour when the layer is first switched on: only the
    gradient from a subject's own words moves its matrix away.
    """

    def __init__(self, subjects: int, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(channels).expand(subjects, channels, channels).clone())

    @classmethod
    def of(cls, subjects: int, channels: int) -> nn.Module:
        """A layer for `subjects` subjects; none (`Shared`) at 0, which keeps single-subject checkpoints' keys."""
        return cls(subjects, channels) if subjects else Shared()

    @override
    def forward(self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"]) -> Float[Tensor, "r c t"]:
        return torch.bmm(self.weight[subject], x)

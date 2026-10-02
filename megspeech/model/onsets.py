"""Word onsets within a span as sinusoids over geometric periods, and the context model's shape."""

import json
import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.objective.heads import Objective

HOLDOUT_GAP = 0.21
"""Median seconds between consecutive vocabulary words in a holdout sentence."""


class Clock(StrEnum):
    """What the transformer is told of each word's position: its onset, or only its rank in the span."""

    ONSET = "onset"
    ORDER = "order"

    def times(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w"]:
        """Seconds to encode; word order is laid out at the holdout's typical spacing so the periods still fit."""
        return onset if self is Clock.ONSET else (mask.cumsum(-1) - 1).to(onset.dtype) * HOLDOUT_GAP


@dataclass(frozen=True)
class ContextConfig:
    width: int = 256
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1
    shortest: float = 0.04
    longest: float = 8.0
    """Onset periods span two 50 Hz samples up to a nine-word span at the holdout's widest gap."""
    timescales: int = 16
    clock: Clock = Clock.ONSET
    objective: Objective = Objective.SOFTMAX

    def save(self, directory: Path) -> Path:
        path = directory / FILE
        path.write_text(json.dumps(asdict(self)))
        return path

    @classmethod
    def load(cls, directory: Path) -> "ContextConfig":
        """The shape a model in `directory` was trained with; models saved before it was recorded used the default."""
        path = directory / FILE
        if not path.exists():
            return CONTEXT
        config = cls(**json.loads(path.read_text()))
        return replace(config, clock=Clock(config.clock), objective=Objective(config.objective))


FILE = "context.json"
CONTEXT = ContextConfig()


class OnsetEncoding(nn.Module):
    """Each word's onset (seconds from the span's first word) as sinusoids over geometric periods."""

    frequency: Tensor

    def __init__(self, config: ContextConfig) -> None:
        super().__init__()
        periods = torch.logspace(math.log10(config.shortest), math.log10(config.longest), config.timescales)
        self.register_buffer("frequency", 2 * math.pi / periods)
        self.project = nn.Linear(2 * config.timescales, config.width)

    @override
    def forward(self, onset: Float[Tensor, "g w"]) -> Float[Tensor, "g w d"]:
        phase = onset[..., None] * self.frequency
        return self.project(torch.cat([phase.sin(), phase.cos()], dim=-1))

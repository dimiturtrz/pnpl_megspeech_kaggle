"""Word onsets within a span as sinusoids over geometric periods, and the context model's shape."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


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

    def save(self, directory: Path) -> Path:
        path = directory / FILE
        path.write_text(json.dumps(asdict(self)))
        return path

    @classmethod
    def load(cls, directory: Path) -> "ContextConfig":
        """The shape a model in `directory` was trained with; models saved before it was recorded used the default."""
        path = directory / FILE
        return cls(**json.loads(path.read_text())) if path.exists() else CONTEXT


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

"""The context model's shape, saved beside every trained model; the per-word encoder it reads and the onset
encoding it builds."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.tree import Tree
from megspeech.model.context.timing import Clock, Intervals
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.conv import CONV, ConvConfig, ConvWords
from megspeech.model.words.megxl import POOLED, RESOLVED, FeatureWords
from megspeech.objective.context import Objective


class Encoder(StrEnum):
    """The CNN trained from scratch, or a head over the features `DataConfig.features` names: MEG-XL's pooled over
    the sensors or each channel's own, or a trained run's CNN embeddings."""

    CONV = "conv"
    CACHED = "cached"
    RESOLVED = "resolved"
    TRAINED = "trained"
    """A head over a trained run's own CNN embeddings (`megspeech.features.trained`), `conv` giving their width."""

    def build(self, config: ConvConfig) -> WordEncoder:
        return BUILDERS[self](config)

    @property
    def cached(self) -> bool:
        """Whether the encoder reads features saved ahead of training rather than the MEG's sensors."""
        return self is not Encoder.CONV


BUILDERS: dict[Encoder, Callable[[ConvConfig], WordEncoder]] = {
    Encoder.CONV: ConvWords,
    Encoder.CACHED: partial(FeatureWords, width=POOLED),
    Encoder.RESOLVED: partial(FeatureWords, width=RESOLVED),
    Encoder.TRAINED: lambda config: FeatureWords(config, config.width),
}


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
    intervals: Intervals = Intervals.NONE
    """Added beside the clock: `log` tells each word its gaps to its neighbours' onsets."""
    objective: Objective = Objective.SOFTMAX
    conv: ConvConfig = CONV
    """The per-word encoder the transformer reads."""
    encoder: Encoder = Encoder.CONV
    """Whether that encoder is the CNN `conv` shapes or a head over saved MEG-XL features, which reads only its
    `dropout`."""
    mixing: int = 0
    """Depthwise kernel, in words, of a convolution module after each transformer layer (`span.SpanConformer`); 0
    keeps the plain transformer. Holdout sentences run 1-9 words, so 3 mixes each word with its neighbours."""
    subjects: int = 0
    final_norm: bool = False
    """A LayerNorm after the pre-norm transformer's last layer, before the context head (plain transformer only)."""

    def __post_init__(self) -> None:
        if self.encoder.cached and self.subjects:
            raise ValueError(
                f"encoder {self.encoder} reads saved features, not sensors, so a per-subject sensor matrix "
                f"(subjects={self.subjects}) has nothing to mix"
            )

    def save(self, directory: Path) -> Path:
        path = directory / FILE
        path.write_text(json.dumps(asdict(self)))
        return path

    @classmethod
    def load(cls, directory: Path) -> "ContextConfig":
        """The shape a model in `directory` was trained with; fields recorded before they existed keep the default."""
        path = directory / FILE
        return Tree(cls).load(json.loads(path.read_text())) if path.exists() else CONTEXT


FILE = "context.json"
CONTEXT = ContextConfig()


class OnsetEncoding(nn.Module):
    """Each word's onset (seconds from the span's first word) as sinusoids over geometric periods."""

    frequency: Tensor

    def __init__(self, config: ContextConfig) -> None:
        super().__init__()
        periods = torch.logspace(math.log10(config.shortest), math.log10(config.longest), config.timescales)
        self.register_buffer("frequency", 2 * math.pi / periods)
        self.project = nn.Linear(2 * config.timescales + config.intervals.width, config.width)

    @override
    def forward(self, onset: Float[Tensor, "g w"], intervals: Float[Tensor, "g w k"]) -> Float[Tensor, "g w d"]:
        """`intervals` are read beside the onset's sinusoids (`Intervals.features`)."""
        phase = onset[..., None] * self.frequency
        return self.project(torch.cat([phase.sin(), phase.cos(), intervals], dim=-1))

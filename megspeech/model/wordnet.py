"""A word classifier over one decimated MEG window: spatial mixing, dilated temporal convolutions, pooling."""

from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import WORDS
from megspeech.model.block import Block
from megspeech.model.conformer import Conformer
from megspeech.model.ssm import Ssm

EPSILON = 1e-6


@dataclass(frozen=True)
class ConvConfig:
    hidden: int = 256
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2)
    dropout: float = 0.3
    kernels: tuple[int, ...] = (3,)
    """Kernel sizes every block sums in parallel at its dilation, a multi-resolution front end: dropping it cost
    BrainWatson's MEGState 5.3 BAcc points (LibriBrain 2025 phoneme track). Odd, so each keeps the time axis."""
    segments: int = 1
    """Equal time chunks the readout pools mean and max over; more keep when in the window the response came."""
    ssm: int = 0
    """Modes per direction of a bidirectional state-space layer after the convolutions; 0 adds none."""
    conformer: int = 0
    """Conformer blocks after the convolutions: self-attention and a 0.5 s depthwise convolution, MEGConformer's
    encoder; with no `dilations` they replace the convolutions."""
    instance: bool = False
    """Z-score each sensor within each window over its recorded samples: MEGConformer (LibriBrain 2025 phoneme
    winner) found per-window normalisation critical for the shift to the holdout."""

    def __post_init__(self) -> None:
        if any(kernel % 2 == 0 for kernel in self.kernels):
            raise ValueError(f"kernels {self.kernels} must be odd to keep the time axis")


CONV = ConvConfig()


class WordNet(nn.Module):
    def __init__(self, config: ConvConfig = CONV) -> None:
        super().__init__()
        self.segments = config.segments
        self.instance = config.instance
        self.width = 2 * config.hidden * config.segments
        self.spatial = nn.Conv1d(CHANNELS, config.hidden, 1)
        self.blocks = nn.ModuleList(Block(config.hidden, d, config.dropout, config.kernels) for d in config.dilations)
        if config.ssm:
            self.blocks.append(Ssm(config.hidden, config.ssm, config.dropout))
        self.blocks.extend(Conformer(config.hidden, config.dropout) for _ in range(config.conformer))
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.width, WORDS))

    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        """`valid` rows alone set the batch statistics; the rest are padding."""
        h = self.spatial(x)
        for block in self.blocks:
            h = block(h, valid)
        pooled = [
            nn.functional.adaptive_avg_pool1d(h, self.segments),
            nn.functional.adaptive_max_pool1d(h, self.segments),
        ]
        return torch.cat([p.flatten(1) for p in pooled], dim=-1)

    def normalise(self, x: Float[Tensor, "b c t"], present: Bool[Tensor, "b 1 t"]) -> Float[Tensor, "b c t"]:
        """`x` z-scored per window and sensor over its `present` samples, zero elsewhere; unchanged unless
        `instance`."""
        if not self.instance:
            return x
        count = present.sum(-1, keepdim=True).clamp(min=1)
        centred = (x - (x * present).sum(-1, keepdim=True) / count) * present
        spread = (centred.square().sum(-1, keepdim=True) / count).sqrt()
        return centred / (spread + EPSILON)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

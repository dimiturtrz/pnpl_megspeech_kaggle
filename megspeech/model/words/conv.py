"""A word classifier over one decimated MEG window: spatial mixing, dilated temporal convolutions, pooling."""

from dataclasses import dataclass
from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.constants import CHANNELS
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.layers.attention_pool import AttentionPool
from megspeech.model.words.layers.conformer import TimeConformer
from megspeech.model.words.layers.dilated import DilatedBlock
from megspeech.model.words.layers.mean_max import MeanMax
from megspeech.model.words.layers.spatial import SpatialAttention
from megspeech.model.words.layers.ssm import Ssm


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
    lora: int = 0
    """Retired: the rank of adapters on an end-to-end MEG-XL trunk, never trained above 0; kept so the configs
    saved with it still load."""
    queries: int = 0
    """Learned attention queries the readout pools time with (`AttentionPool`); 0 pools mean and max over
    `segments` chunks. Two queries scored 0.615 vs 0.621 mean-max on one seed (megspeech-7gf.45)."""
    merger: int = 0
    """Channels a `SpatialAttention` merges the sensors into before the 1x1 spatial convolution, with spatial
    dropout in training: Défossez et al. 2023's front end, 270 there; 0 mixes the sensors with the convolution alone."""
    gated: bool = False
    """GLU-gate every dilated block in place of its GELU (`DilatedBlock`), d'Ascoli et al.'s trunk activation."""

    def __post_init__(self) -> None:
        if any(kernel % 2 == 0 for kernel in self.kernels):
            raise ValueError(f"kernels {self.kernels} must be odd to keep the time axis")
        if self.lora:
            raise ValueError(f"lora={self.lora}: the end-to-end MEG-XL encoder it adapted is retired")

    @property
    def width(self) -> int:
        """The embedding's width: each attention query's pooled channels, or each segment's mean and max."""
        return self.hidden * (self.queries or 2 * self.segments)


CONV = ConvConfig()


class ConvWords(WordEncoder):
    def __init__(self, config: ConvConfig = CONV, channels: int = CHANNELS) -> None:
        """Words from recordings of `channels` rows; a `SpatialAttention` front end places the 306 sensors, so it
        reads nothing else."""
        if config.merger and channels != CHANNELS:
            raise ValueError(f"a {config.merger}-channel spatial merger lays out {CHANNELS} sensors, not {channels}")
        super().__init__(config.width, instance=config.instance)
        self.pool = AttentionPool(config.hidden, config.queries) if config.queries else MeanMax(config.segments)
        self.spatial = (
            nn.Sequential(SpatialAttention(config.merger), nn.Conv1d(config.merger, config.hidden, 1))
            if config.merger
            else nn.Conv1d(channels, config.hidden, 1)
        )
        self.blocks = nn.ModuleList(
            DilatedBlock(config.hidden, d, config.dropout, config.kernels, gated=config.gated) for d in config.dilations
        )
        if config.ssm:
            self.blocks.append(Ssm(config.hidden, config.ssm, config.dropout))
        self.blocks.extend(TimeConformer(config.hidden, config.dropout) for _ in range(config.conformer))
        self.head = self.classifier(config.dropout)

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        h = self.spatial(x)
        for block in self.blocks:
            h = block(h, valid)
        return self.pool(h)

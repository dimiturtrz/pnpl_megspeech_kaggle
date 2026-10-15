"""A word classifier over one decimated MEG window: spatial mixing, dilated temporal convolutions, pooling."""

from dataclasses import dataclass
from typing import ClassVar, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from core.tree import Retired
from megspeech.data.constants import CHANNELS
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.heads import WordHead
from megspeech.model.words.layers.conformer import TimeConformer
from megspeech.model.words.layers.dilated import DilatedBlock
from megspeech.model.words.layers.downsample import Downsample
from megspeech.model.words.layers.readouts.attention_pool import AttentionPool
from megspeech.model.words.layers.readouts.mean_max import MeanMax
from megspeech.model.words.layers.sites.sensorwise import Grouping, Sensorwise
from megspeech.model.words.layers.sites.spatial import SpatialAttention
from megspeech.model.words.pyramid import PYRAMID, PyramidConfig


@dataclass(frozen=True)
class ConvConfig:
    RETIRED: ClassVar[dict[str, Retired]] = {
        "kernels": Retired((3,), "multi-kernel blocks (kernels) are retired, killed by megspeech-7gf.26"),
        "ssm": Retired(0, "the state-space layer (ssm) is retired, killed by megspeech-7gf.27"),
        "lora": Retired(0, "lora adapted the retired end-to-end MEG-XL encoder and never trained above 0"),
        "gated": Retired(default=False, reason="the GLU-gated trunk (gated) is retired, killed by megspeech-7gf.29.7"),
    }

    hidden: int = 256
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2)
    dropout: float = 0.3
    segments: int = 1
    """Equal time chunks the readout pools mean and max over; more keep when in the window the response came."""
    conformer: int = 0
    """Conformer blocks after the convolutions: self-attention and a 0.5 s depthwise convolution, MEGConformer's
    encoder; with no `dilations` they replace the convolutions."""
    instance: bool = False
    """Z-score each sensor within each window over its recorded samples: MEGConformer (LibriBrain 2025 phoneme
    winner) found per-window normalisation critical for the shift to the Kaggle test."""
    queries: int = 0
    """Learned attention queries the readout pools time with (`AttentionPool`); 0 pools mean and max over
    `segments` chunks. Two queries scored 0.615 vs 0.621 mean-max on one seed (megspeech-7gf.45)."""
    merger: int = 0
    """Channels a `SpatialAttention` merges the sensors into before the 1x1 spatial convolution, with spatial
    dropout in training: Défossez et al. 2023's front end, 270 there; 0 mixes the sensors with the convolution alone."""
    sensorwise: int = 0
    """Channels per sensor of a temporal stack shared by every sensor, run before the sensors are fused (`Sensorwise`):
    time first, space second, weights shared across sensors; 0 mixes the sensors first."""
    sensorwise_depth: int = 0
    """Dilated blocks in the per-sensor stack, the first `sensorwise_depth` of `dilations`; 0 = all."""
    sensorwise_site: int = 0
    """Sensors grouped per site, the triplet entering together as `sensorwise_site` channels
    (3 = Neuromag's magnetometer and two gradiometers per site, contiguous in the 306); 0 = each sensor alone."""
    sensorwise_out: int = 0
    """Channels each site (or sensor) is reduced to by a 1x1 convolution before fusion, shrinking the fuse layer;
    0 keeps `sensorwise`."""
    sensorwise_attend: int = 0
    """Pre-norm transformer layers across the sites (after the site embedding, per time step); 0 = none."""
    sensorwise_heads: int = 2
    """Attention heads of the site-axis layers; must divide the site width."""
    sensorwise_pool: int = 0
    """Learned queries cross-attending over the site tokens in place of the flatten fuse (`SetPool`), layout-agnostic
    and maskable; 0 = flatten."""
    sensorwise_drop: float = 0.0
    """Training-only share of sites dropped per window (masked out of the pool's attention, or zeroed and the
    survivors rescaled under the flatten fuse); 0 = none."""
    head: WordHead = WordHead.LINEAR
    """How the embedding is read out as the 50 words: one linear layer, or one per part of speech (`TaggedHead`)."""
    stages: tuple[int, ...] = ()
    """Indices into `dilations` before which the trunk halves time and widens by `growth` (`Downsample`): the
    AlexNet/ResNet pyramid; empty keeps every block at full rate and width."""
    growth: float = 1.5
    """Channel multiplier per stage, each width rounded to a multiple of 8."""
    pyramid: PyramidConfig = PYRAMID
    """A sentence model's intermediate-depth readout of these blocks' maps: a norm-bounded correction to both heads'
    logits, everything under it frozen (`model.pyramid`); no taps reads none."""

    def __post_init__(self) -> None:
        if self.merger and self.sensorwise:
            raise ValueError(f"merger={self.merger} and sensorwise={self.sensorwise} are two spatial front ends")
        if self.sensorwise_depth > len(self.dilations):
            raise ValueError(f"sensorwise_depth={self.sensorwise_depth} exceeds the {len(self.dilations)} dilations")
        self._check_sensorwise()
        if list(self.stages) != sorted(set(self.stages)) or any(
            not 1 <= stage < len(self.dilations) for stage in self.stages
        ):
            raise ValueError(f"stages {self.stages} must increase strictly within 1..{len(self.dilations) - 1}")
        if self.growth < 1:
            raise ValueError(f"growth={self.growth} must be at least 1")

    def _check_sensorwise(self) -> None:
        for name in ("site", "out", "attend", "pool", "drop"):
            value = getattr(self, f"sensorwise_{name}")
            if value and not self.sensorwise:
                raise ValueError(f"sensorwise_{name}={value} needs sensorwise")
        if not 0 <= self.sensorwise_drop < 1:
            raise ValueError(f"sensorwise_drop={self.sensorwise_drop} must be in [0, 1)")
        if (self.sensorwise_attend or self.sensorwise_pool) and (
            self.sensorwise_out or self.sensorwise
        ) % self.sensorwise_heads:
            raise ValueError(f"sensorwise_heads={self.sensorwise_heads} must divide the site width")

    @property
    def widths(self) -> tuple[int, ...]:
        """The trunk's width after 0..len(dilations) blocks: `hidden`, grown once per stage reached."""
        return (
            self.hidden,
            *(self.grown(sum(stage <= i for stage in self.stages)) for i in range(len(self.dilations))),
        )

    def grown(self, stages: int) -> int:
        return round(self.hidden * self.growth**stages / WIDTH_MULTIPLE) * WIDTH_MULTIPLE if stages else self.hidden

    @property
    def out(self) -> int:
        """The trunk's final width, what the pool and every layer after the convolutions work in."""
        return self.widths[-1]

    @property
    def width(self) -> int:
        """The embedding's width: each attention query's pooled channels, or each segment's mean and max."""
        return self.out * (self.queries or 2 * self.segments)


WIDTH_MULTIPLE = 8
CONV = ConvConfig()


class ConvWords(WordEncoder):
    def __init__(self, config: ConvConfig = CONV, channels: int = CHANNELS) -> None:
        """Words from recordings of `channels` rows; a `SpatialAttention` front end places the 306 sensors, so it
        reads nothing else."""
        if config.merger and channels != CHANNELS:
            raise ValueError(f"a {config.merger}-channel spatial merger lays out {CHANNELS} sensors, not {channels}")
        super().__init__(config.width, instance=config.instance)
        self.hidden = config.out
        self.pool = AttentionPool(config.out, config.queries) if config.queries else MeanMax(config.segments)
        self.spatial = self.front(config, channels)
        self.blocks = nn.ModuleList()
        for i, d in enumerate(config.dilations):
            if i in config.stages:
                self.blocks.append(Downsample(config.widths[i], config.widths[i + 1]))
            self.blocks.append(DilatedBlock(config.widths[i + 1], d, config.dropout))
        self.blocks.extend(TimeConformer(config.out, config.dropout) for _ in range(config.conformer))
        self.depths = (*config.widths[1:], *(config.out,) * config.conformer)
        self.head = self.classifier(config.dropout, config.head)

    @staticmethod
    def front(config: ConvConfig, channels: int) -> nn.Module:
        """The layer taking the sensors to `hidden` channels: a per-sensor stack then fusion, a spatial merger, or one
        1x1 convolution."""
        if config.sensorwise:
            stack = (
                DilatedBlock(config.sensorwise, d, config.dropout)
                for d in config.dilations[: config.sensorwise_depth or None]
            )
            return Sensorwise(
                channels,
                config.sensorwise,
                config.hidden,
                stack,
                Grouping(
                    config.sensorwise_site or 1,
                    config.sensorwise_out,
                    config.sensorwise_attend,
                    config.sensorwise_heads,
                    config.dropout,
                    config.sensorwise_pool,
                    config.sensorwise_drop,
                ),
            )
        if config.merger:
            return nn.Sequential(SpatialAttention(config.merger), nn.Conv1d(config.merger, config.hidden, 1))
        return nn.Conv1d(channels, config.hidden, 1)

    def taps(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> list[Float[Tensor, "b h t"]]:
        """Every block's per-timestep feature map, the last the one `embed` pools."""
        h = self.spatial(x, valid) if isinstance(self.spatial, Sensorwise) else self.spatial(x)
        maps = []
        for block in self.blocks:
            h = block(h, valid)
            if not isinstance(block, Downsample):
                maps.append(h)
        return maps

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        return self.pool(self.taps(x, valid)[-1])

    @override
    def time_width(self) -> int:
        return self.hidden

    @override
    def embed_timed(
        self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, "b h s"]]:
        times = self.taps(x, valid)[-1]
        return self.pool(times), times

    @override
    def tap_widths(self) -> tuple[int, ...]:
        return self.depths

    @override
    def embed_tapped(
        self, x: Float[Tensor, "b c t"], taps: tuple[int, ...], levels: tuple[int, ...], valid: Bool[Tensor, " b"]
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, "b k"]]:
        maps = self.taps(x, valid)
        pooled = [nn.functional.adaptive_avg_pool1d(maps[tap], level).flatten(1) for tap in taps for level in levels]
        return self.pool(maps[-1]), torch.cat(pooled, dim=-1)

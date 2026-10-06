"""What the student is not shown: time patches of its window and whole sensors."""

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Channels(StrEnum):
    DROPOUT = "dropout"
    """Random sensors zeroed after the model's scaling and whitening, which mixes sensors, so a neighbour's trace can
    still be read off the rest."""
    REGION = "region"
    """One contiguous patch of sensors zeroed on the raw recording, before scaling and whitening: the field there has
    to be inferred from other regions (Défossez et al. 2023's regional dropout)."""


@dataclass(frozen=True)
class Region:
    """For every sensor, itself and its nearest neighbours by position."""

    nearest: Int[Tensor, "c n"] = field(default_factory=lambda: torch.zeros(0, 0, dtype=torch.long))

    @classmethod
    def of(cls, positions: Float[Tensor, "c 3"], size: int) -> "Region":
        """Each sensor with its `size` nearest by 3D distance; the magnetometer and gradiometers that share its
        position count as neighbours."""
        distance = torch.cdist(positions, positions).fill_diagonal_(-1.0)
        return cls(distance.topk(min(size + 1, len(positions)), dim=1, largest=False).indices)

    def silent(self, rows: int, generator: torch.Generator) -> Bool[Tensor, "b c"]:
        """Per window one random seed sensor and its neighbours."""
        seeds = torch.randint(self.nearest.shape[0], (rows,), generator=generator).to(self.nearest.device)
        silent = torch.zeros(rows, self.nearest.shape[0], dtype=torch.bool, device=self.nearest.device)
        return silent.scatter_(1, self.nearest[seeds], value=True)


@dataclass(frozen=True)
class Masking:
    ratio: float = 0.4
    """Share of the window's time patches the student does not see: data2vec masks 0.65 of speech frames, wav2vec 2.0
    0.5 of them."""
    patch: float = 0.05
    """Seconds one masked patch spans, about one 50 ms response component."""
    channels: Channels = Channels.DROPOUT
    dropout: float = 0.1
    """Share of sensors zeroed in each window under `Channels.DROPOUT`, a light channel dropout."""
    region: int = 30
    """Neighbours zeroed with the seed sensor under `Channels.REGION`: 31 of 306 sensors, about the share `dropout`
    zeroes and the order of Défossez et al.'s 0.2-radius patch."""

    def __post_init__(self) -> None:
        if not 0 < self.ratio < 1:
            raise ValueError(f"mask ratio {self.ratio} is not in (0, 1)")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"channel dropout {self.dropout} is not in [0, 1)")
        if self.region < 1:
            raise ValueError(f"region of {self.region} neighbours masks nothing")

    def patches(self, rate: float) -> int:
        """Samples per patch at `rate`, at least one."""
        return max(1, round(self.patch * rate))

    def draw(self, rows: int, samples: int, rate: float, generator: torch.Generator) -> Bool[Tensor, "b t"]:
        """Each of `rows` windows' masked samples: `ratio` of its patches, exactly, chosen by rank of a uniform draw."""
        width = self.patches(rate)
        count = -(-samples // width)
        chosen = max(1, round(self.ratio * count))
        ranks = torch.rand(rows, count, generator=generator).argsort(dim=1).argsort(dim=1)
        return (ranks < chosen).repeat_interleave(width, dim=1)[:, :samples]

    def silence(self, x: Float[Tensor, "b c t"], region: Region, generator: torch.Generator) -> Float[Tensor, "b c t"]:
        """The raw windows `x` with a patch of sensors zeroed under `Channels.REGION`; as they are otherwise."""
        if self.channels is not Channels.REGION:
            return x
        return x * ~region.silent(len(x), generator)[..., None]

    def apply(
        self,
        x: Float[Tensor, "b c t"],
        hidden: Bool[Tensor, "b t"],
        token: Float[Tensor, "c 1"],
        generator: torch.Generator,
    ) -> Float[Tensor, "b c t"]:
        """`x` with the `hidden` samples replaced by the learned `token` and, under `Channels.DROPOUT`, a share of
        sensors zeroed."""
        dropout = self.dropout if self.channels is Channels.DROPOUT else 0.0
        kept = torch.rand(x.shape[:2], generator=generator).ge(dropout).to(x.device)[..., None]
        return torch.where(hidden[:, None, :], token.to(x.dtype), x) * kept

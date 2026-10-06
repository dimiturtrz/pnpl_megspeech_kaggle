"""A channel-first front end: one temporal stack shared by every sensor, the sensors fused after."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.words.layers.dilated import DilatedBlock


@dataclass(frozen=True)
class Grouping:
    site: int = 1
    """Contiguous channels entering together as one site; 1 = each sensor alone."""
    out: int = 0
    """Channels each site is reduced to before fusion; 0 keeps the stack's width."""


class Sensorwise(nn.Module):
    """Time first, space second: each site's `site` contiguous channels (1 = each sensor alone, 3 = a Neuromag
    magnetometer and its two gradiometers) run together through the same dilated `stack` (weights shared across
    sites), an optional 1x1 convolution reduces each site's `width` to `out`, a learned per-site embedding names
    the site, and a 1x1 convolution fuses all `channels // site * out` features into `hidden`."""

    def __init__(
        self, channels: int, width: int, hidden: int, stack: Iterable[DilatedBlock], grouping: Grouping | None = None
    ) -> None:
        super().__init__()
        grouping = grouping or Grouping()
        site = grouping.site
        if channels % site:
            raise ValueError(f"{channels} channels do not split into sites of {site}")
        self.site = site
        sites = channels // site
        self.lift = nn.Conv1d(site, width, 1)
        self.stack = nn.ModuleList(stack)
        out = grouping.out or width
        self.reduce = nn.Conv1d(width, out, 1) if out != width else nn.Identity()
        self.sensor = nn.Parameter(torch.zeros(sites, out, 1))
        self.fuse = nn.Conv1d(sites * out, hidden, 1)

    def features(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b s w t"]:
        """Every site's temporal features, before the site embedding; row-wise over sites, so equivariant to
        their order. `valid` windows set the batch statistics, each repeated over its sites."""
        b, c, t = x.shape
        sites = c // self.site
        h = self.lift(x.reshape(b * sites, self.site, t))
        rows = None if valid is None else valid.repeat_interleave(sites)
        for block in self.stack:
            h = block(h, rows)
        return h.reshape(b, sites, -1, t)

    @override
    def forward(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        h = self.features(x, valid)
        b, sites, width, t = h.shape
        h = self.reduce(h.reshape(b * sites, width, t)).reshape(b, sites, -1, t) + self.sensor
        return self.fuse(h.flatten(1, 2))

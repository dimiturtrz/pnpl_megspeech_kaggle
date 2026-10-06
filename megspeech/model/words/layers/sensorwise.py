"""A channel-first front end: one temporal stack shared by every sensor, the sensors fused after."""

from collections.abc import Iterable
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.words.layers.dilated import DilatedBlock


class Sensorwise(nn.Module):
    """Time first, space second: each sensor's trace runs alone through the same dilated `stack`
    (weights shared across sensors), a learned per-sensor embedding names the sensor, and a 1x1 convolution fuses
    all `channels * width` features into `hidden`."""

    def __init__(self, channels: int, width: int, hidden: int, stack: Iterable[DilatedBlock]) -> None:
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.stack = nn.ModuleList(stack)
        self.sensor = nn.Parameter(torch.zeros(channels, width, 1))
        self.fuse = nn.Conv1d(channels * width, hidden, 1)

    def features(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b c w t"]:
        """Every sensor's temporal features, before the sensor embedding; row-wise over sensors, so equivariant to
        their order. `valid` windows set the batch statistics, each repeated over its sensors."""
        b, c, t = x.shape
        h = self.lift(x.reshape(b * c, 1, t))
        rows = None if valid is None else valid.repeat_interleave(c)
        for block in self.stack:
            h = block(h, rows)
        return h.reshape(b, c, -1, t)

    @override
    def forward(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        h = self.features(x, valid) + self.sensor
        return self.fuse(h.flatten(1, 2))

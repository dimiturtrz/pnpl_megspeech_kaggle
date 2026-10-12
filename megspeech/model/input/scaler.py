"""Robust per-channel scaling of MEG windows, fit on the training split."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

CLAMP = 5.0
"""d'Ascoli 2025 clamp after robust scaling: a window past 5 IQR is an artefact, not a word."""

SCALER_SAMPLE = 20_000
QUANTILE_SAMPLE = 100_000


class Scaler(nn.Module):
    """Per-channel median / interquartile range from training windows, then a clamp."""

    median: Tensor
    spread: Tensor

    def __init__(self, train: Float[Tensor, "n c t"]) -> None:
        super().__init__()
        median, spread = self.quartiles(train[torch.randperm(len(train), device=train.device)[:SCALER_SAMPLE]])
        self.register_buffer("median", median)
        self.register_buffer("spread", spread)

    @staticmethod
    def quartiles(sample: Float[Tensor, "n c t"]) -> tuple[Float[Tensor, "c 1"], Float[Tensor, "c 1"]]:
        """Each channel's median and interquartile range over up to `QUANTILE_SAMPLE` of `sample`'s points."""
        flat = sample.float().transpose(0, 1).reshape(sample.shape[1], -1)
        flat = flat[:, :: max(1, flat.shape[1] // QUANTILE_SAMPLE)]
        levels = torch.tensor([0.25, 0.5, 0.75], device=flat.device)
        low, median, high = torch.quantile(flat, levels, dim=1)
        return median[:, None], (high - low)[:, None]

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c t"]:
        """Each channel in interquartile ranges from its median; a flat channel's zero range divides as the smallest
        normal float, so it saturates at the clamp instead of reading NaN."""
        spread = self.spread.clamp(min=torch.finfo(self.spread.dtype).tiny)
        return ((x.float() - self.median) / spread).clamp(-CLAMP, CLAMP)

"""Train-time perturbations of word windows, applied on the device per batch."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class Augment:
    """Whole-sensor dropout, per-sensor log-normal gain and onset jitter; all off by default."""

    dropout: float = 0.0
    gain: float = 0.0
    jitter: int = 0
    """Samples each training window's start moves, uniform in [-jitter, jitter]: forced-alignment onsets err by
    tens of milliseconds, one or two samples at 50 Hz."""

    def __call__(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        shape = (*x.shape[:2], 1)
        keep = torch.rand(shape, device=x.device) >= self.dropout
        scale = torch.exp(self.gain * torch.randn(shape, device=x.device))
        return x * (keep * scale).to(x.dtype)

"""A residual bidirectional diagonal state-space layer over hidden MEG features (S4D-Real, Gu et al. 2022).

Each hidden channel runs `state` real decaying modes forward and `state` backward in time; their summed impulse
responses form one long convolution kernel applied by FFT, so a window's every sample reaches every other one."""

import math
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.norm import MaskedBatchNorm1d

SHORTEST = 0.01
LONGEST = 1.0
"""Step sizes, in samples, the modes' decay rates are scaled by: at 50 Hz the slowest mode then decays over ~2 s
and the fastest within one sample."""


class Ssm(nn.Module):
    def __init__(self, width: int, state: int, dropout: float) -> None:
        super().__init__()
        rates = torch.log(0.5 + torch.arange(state, dtype=torch.float32)).expand(2, width, state)
        self.rate = nn.Parameter(rates.clone())
        self.output = nn.Parameter(torch.randn(2, width, state) / math.sqrt(state))
        steps = torch.rand(width) * (math.log(LONGEST) - math.log(SHORTEST)) + math.log(SHORTEST)
        self.step = nn.Parameter(steps)
        self.skip = nn.Parameter(torch.ones(width))
        self.body = nn.Sequential(MaskedBatchNorm1d(width), nn.GELU(), nn.Dropout(dropout))

    def kernel(self, length: int) -> Float[Tensor, "2 h l"]:
        """The forward and backward impulse responses over `length` samples."""
        step = self.step.exp()[None, :, None]
        decay = -self.rate.exp() * step
        lags = torch.arange(length, device=decay.device, dtype=decay.dtype)
        return torch.einsum("dhn,dhnl->dhl", self.output * step, (decay[..., None] * lags).exp())

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """`valid` rows alone set the batch statistics; the rest are padding. The convolution runs in float32 even
        under autocast, which FFTs do not support at half precision."""
        length = x.shape[-1]
        size = 2 * length
        with torch.autocast(x.device.type, enabled=False):
            forward, backward = torch.fft.rfft(self.kernel(length), n=size)
            signal = x.float()
            ahead = torch.fft.irfft(torch.fft.rfft(signal, n=size) * forward, n=size)[..., :length]
            behind = torch.fft.irfft(torch.fft.rfft(signal.flip(-1), n=size) * backward, n=size)[..., :length]
            mixed = (ahead + behind.flip(-1) + self.skip[:, None] * signal).to(x.dtype)
        norm, act, drop = self.body
        return x + drop(act(norm(mixed, valid)))

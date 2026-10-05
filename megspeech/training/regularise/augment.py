"""Train-time perturbations of word windows, applied on the device per batch."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class Augment:
    """Whole-sensor dropout, per-sensor log-normal gain, onset jitter, time and band masks and additive noise; all
    off by default. Masks and noise leave each window's onset where it is, so the time-locked response survives."""

    dropout: float = 0.0
    gain: float = 0.0
    jitter: int = 0
    """Samples each training window's start moves, uniform in [-jitter, jitter]: forced-alignment onsets err by
    tens of milliseconds, one or two samples at 50 Hz."""
    mask: int = 0
    """Samples of one span per window, at a uniform start, zeroed on every sensor."""
    band: int = 0
    """Frequency bins of one band per window, at a uniform lowest bin, zeroed on every sensor."""
    noise: float = 0.0
    """Gaussian noise added to each sensor, as a share of that sensor's spread over the window."""
    mix: float = 0.0
    """Largest share of another window, circularly shifted off its own onsets, added as background (BGMix without its
    label); each window draws its share uniformly up to it. Real MEG, so it stays in the field's own subspace."""
    bgmix: float = 0.0
    """Probability a labelled training word's window becomes its (subject, label) template plus another labelled word's
    residual (`megspeech.training.regularise.bgmix`, trial-level BGMix), the label kept; 0 mixes none."""
    spatial: float = 0.0
    """Radius of the sensor patch each window loses (`Regional`), as a share of the array's widest extent."""

    def __call__(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        """The perturbed windows in float32: the half-precision cache's largest values sit near its overflow."""
        x = x.float()
        shape = (*x.shape[:2], 1)
        keep = torch.rand(shape, device=x.device) >= self.dropout
        scale = torch.exp(self.gain * torch.randn(shape, device=x.device))
        x = x * (keep * scale)
        x = self.masked(x, self.mask, x.shape[-1])
        if self.band:
            spectrum = self.masked(torch.fft.rfft(x, dim=-1), self.band, x.shape[-1] // 2 + 1)
            x = torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)
        if self.noise:
            spread = x.std(dim=-1, keepdim=True)
            x = x + self.noise * spread * torch.randn_like(x) * (x != 0)
        return self.mixed(x)

    def mixed(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        """`x` plus a uniform share up to `mix` of a shuffled window, rolled by a uniform shift; padding stays zero."""
        if not self.mix:
            return x
        other = x[torch.randperm(x.shape[0], device=x.device)]
        shift = torch.randint(0, x.shape[-1], (x.shape[0], 1, 1), device=x.device)
        index = (torch.arange(x.shape[-1], device=x.device) + shift) % x.shape[-1]
        background = other.gather(-1, index.expand_as(other))
        share = self.mix * torch.rand(x.shape[0], 1, 1, device=x.device)
        return x + share * background * (x != 0)

    @staticmethod
    def masked(x: Float[Tensor, "n c f"], width: int, length: int) -> Float[Tensor, "n c f"]:
        """`x` with `width` consecutive positions along its last axis zeroed, at a start drawn per window."""
        if not width:
            return x
        start = torch.randint(0, max(length - width, 0) + 1, (x.shape[0], 1), device=x.device)
        position = torch.arange(x.shape[-1], device=x.device)
        hidden = (position >= start) & (position < start + width)
        return x.masked_fill(hidden[:, None, :], 0)

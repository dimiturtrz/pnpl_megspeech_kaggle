"""Train-time perturbations of word windows, applied on the device per batch."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.training.regularise.inputs.bgmix import BGMixKey


@dataclass(frozen=True)
class Augment:
    """Whole-sensor dropout, per-sensor log-normal gain, onset jitter, time and band masks and additive noise; all
    off by default. Masks and noise leave each window's onset where it is, so the time-locked response survives."""

    dropout: float = 0.0
    gain: float = 0.0
    jitter: int = 0
    """Samples each training window's start moves, uniform in [-jitter, jitter]: forced-alignment onsets err by
    tens of milliseconds, one or two samples at 50 Hz."""
    stretch: float = 0.0
    """Largest relative change of each training window's time scale around its word's onset, which stays put: a factor
    per window, uniform in [1 - stretch, 1 + stretch] (`megspeech.training.regularise.inputs.stretch`); 0 stretches
    none."""
    mask: int = 0
    """Samples of one span per window, at a uniform start, zeroed on every sensor."""
    band: int = 0
    """Frequency bins of one band per window, at a uniform lowest bin, zeroed on every sensor."""
    chance: float = 1.0
    """Probability each window gets its time mask and its band mask, drawn independently per window and per mask."""
    noise: float = 0.0
    """Gaussian noise added to each sensor, as a share of that sensor's spread over the window."""
    mix: float = 0.0
    """Largest share of another window, circularly shifted off its own onsets, added as background (BGMix without its
    label); each window draws its share uniformly up to it. Real MEG, so it stays in the field's own subspace."""
    bgmix: float = 0.0
    """Probability a labelled training word's window becomes its (`bgmix_key`, label) template plus another labelled
    word's residual (`megspeech.training.regularise.inputs.bgmix`, trial-level BGMix), the label kept; 0 mixes none."""
    bgmix_key: BGMixKey = BGMixKey.SUBJECT
    """What a BGMix template is per word: its subject, or the dataset it was recorded in (`BGMixKey`)."""
    average: int = 1
    """Windows of the same (subject, label) each labelled training word's window is averaged with
    (`megspeech.training.regularise.inputs.average`, MEGConformer's dynamic grouping), itself included; 1 averages
    none."""
    average_chance: float = 1.0
    """Probability a labelled training word's window is averaged, when `average` is above 1."""
    spatial: float = 0.0
    """Radius of the sensor patch each window loses (`Regional`), as a share of the array's widest extent."""
    right_cut: float = 0.0
    """Share of labelled training words that end their span, reached by cutting random spans short after a random word
    (`megspeech.training.regularise.inputs.rightcut`); the Kaggle test's is 156 of its 868 rows in multi-row epochs,
    0.18. 0 cuts none."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.chance <= 1.0:
            raise ValueError(f"chance must lie in [0, 1], got {self.chance}")
        if not 0.0 <= self.stretch < 1.0:
            raise ValueError(f"stretch must lie in [0, 1), got {self.stretch}")
        if not 0.0 <= self.right_cut < 1.0:
            raise ValueError(f"right_cut must lie in [0, 1), got {self.right_cut}")
        if self.average < 1 or not 0.0 <= self.average_chance <= 1.0:
            raise ValueError(
                f"average must be >= 1, average_chance in [0, 1]; got {self.average}, {self.average_chance}"
            )

    def __call__(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        """The perturbed windows in float32: the half-precision cache's largest values sit near its overflow."""
        x = self.sensors(x.float())
        x = self.masked(x, self.mask, x.shape[-1], self.chance)
        if self.band:
            spectrum = self.masked(torch.fft.rfft(x, dim=-1), self.band, x.shape[-1] // 2 + 1, self.chance)
            x = torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)
        if self.noise:
            spread = x.std(dim=-1, keepdim=True)
            x = x + self.noise * spread * torch.randn_like(x) * (x != 0)
        return self.mixed(x)

    @cached_property
    def sensors(self) -> Callable[[Float[Tensor, "n c t"]], Float[Tensor, "n c t"]]:
        """The sensor dropout and gain stage, picked once: the identity when both are 0, whose draws would keep every
        sensor at gain 1."""
        return self.scaled if self.dropout or self.gain else self.unscaled

    def scaled(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        shape = (*x.shape[:2], 1)
        keep = torch.rand(shape, device=x.device) >= self.dropout
        scale = torch.exp(self.gain * torch.randn(shape, device=x.device))
        return x * (keep * scale)

    @staticmethod
    def unscaled(x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        return x

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
    def masked(x: Float[Tensor, "n c f"], width: int, length: int, chance: float = 1.0) -> Float[Tensor, "n c f"]:
        """`x` with `width` consecutive positions along its last axis zeroed, at a start drawn per window, in each
        window with probability `chance`."""
        if not width:
            return x
        start = torch.randint(0, max(length - width, 0) + 1, (x.shape[0], 1), device=x.device)
        position = torch.arange(x.shape[-1], device=x.device)
        hidden = (position >= start) & (position < start + width)
        hidden = hidden & (torch.rand(x.shape[0], 1, device=x.device) < chance)
        return x.masked_fill(hidden[:, None, :], 0)

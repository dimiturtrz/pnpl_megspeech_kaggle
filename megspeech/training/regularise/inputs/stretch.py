"""Train-time time stretch (Brain2Qwerty v2, arXiv 2608.18114): each training word window resampled in time by its own
factor drawn uniformly in [1 - share, 1 + share], around its word's onset sample, which stays where it is.

Onset jitter (`Augment.jitter`, megspeech-7gf.18) shifts the whole window, onset included; a stretch keeps the onset
locked and moves each later component in proportion to its latency, the trial-to-trial spread of late responses. Each
word's window is stretched around its own onset, so the span's onsets, which the context model reads as timing, do not
move; a continuous segment batch, whose words sit at their onsets inside one row, is not stretched (`RunConfig` rules).
"""

from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from megspeech.data.config import DataConfig


@dataclass(frozen=True)
class Stretch:
    share: float = 0.0
    """Largest relative change of each window's time scale; 0 stretches none."""
    pivot: int = 0
    """The word onset's sample in its window: minus the window's offset from the onset."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.share < 1.0:
            raise ValueError(f"stretch must lie in [0, 1), got {self.share}")

    @classmethod
    def of(cls, share: float, data: DataConfig) -> Self:
        """Stretches of up to `share` around the onset of windows cut as `data` reads them."""
        return cls(share, -data.cut().offset)

    def __call__(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        """Each window in float32, read at its own time scale: output sample j is input sample pivot + (j - pivot) / r,
        linearly interpolated, zero past the window's ends."""
        if not self.share:
            return x
        factor = 1.0 + self.share * (2.0 * torch.rand(x.shape[0], 1, device=x.device) - 1.0)
        return self.resampled(x, factor)

    def resampled(self, x: Float[Tensor, "n c t"], factor: Float[Tensor, "n 1"]) -> Float[Tensor, "n c t"]:
        """`x` with window i's time scale multiplied by `factor[i]` around `pivot`."""
        samples = x.shape[-1]
        position = torch.arange(samples, device=x.device, dtype=torch.float32)
        source = self.pivot + (position - self.pivot) / factor
        low = source.floor()
        weight = (source - low)[:, None, :]
        padded = F.pad(x.float(), (1, 1))
        index = (low.long() + 1)[:, None, :].expand(-1, x.shape[1], -1)
        below = padded.gather(-1, index.clamp(0, samples + 1))
        above = padded.gather(-1, (index + 1).clamp(0, samples + 1))
        return torch.lerp(below, above, weight)

"""Word windows cut on demand from one continuous device signal: a single gather, no per-word loop.

A word's window starts `offset` samples after its onset sample `at` and runs `length` samples, at the signal's
50 Hz `RATE` (seconds x `RATE`). Each window then has every channel's mean over the window removed, as the
holdout's windows do (`megspeech.data.epochs.Window`).
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.continuous import RATE


@dataclass(frozen=True)
class Windows:
    """Word `i`'s window is `meg[:, at[i] + offset : at[i] + offset + length]`, channel-demeaned."""

    meg: Float[Tensor, "c t"]
    at: Int[Tensor, " n"]
    length: int = RATE
    offset: int = 0

    @classmethod
    def stacked(cls, x: Float[Tensor, "n c s"]) -> "Windows":
        """Windows already cut, laid end to end so each one is its own word's window."""
        count, channels, samples = x.shape
        meg = x.transpose(0, 1).reshape(channels, count * samples)
        return cls(meg, torch.arange(count, device=x.device) * samples, samples)

    def __len__(self) -> int:
        return len(self.at)

    @property
    def device(self) -> torch.device:
        return self.meg.device

    def __call__(self, words: Int[Tensor, " k"]) -> Float[Tensor, "k c s"]:
        """The windows of `words`, in the signal's dtype; centring runs in float32 so float16 only rounds once."""
        steps = torch.arange(self.length, device=self.meg.device)
        cut = self.meg[:, self.at[words, None] + self.offset + steps].transpose(0, 1).float()
        return (cut - cut.mean(dim=-1, keepdim=True)).to(self.meg.dtype)

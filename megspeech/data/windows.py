"""Word windows cut on demand from one continuous device signal: a single gather, no per-word loop.

A word's window starts `offset` samples after its onset sample `at` and runs `length` samples, at the signal's
50 Hz `RATE` (seconds x `RATE`). It stops at the word's exclusive bound `end`: samples from `end` on are zero, so
a window longer than 1 s never reads past the stretch the holdout would hold for that word. Each window then has
every channel's mean over its valid samples removed. Training, validation and holdout windows all come through
here, from signals the same `continuous.Decimator` cached.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.epochs import RATE


@dataclass(frozen=True)
class Windows:
    """Word `i`'s window is `meg[:, at[i] + offset : at[i] + offset + length]`, zero from `end[i]` on, centred on
    its valid samples."""

    meg: Float[Tensor, "c t"]
    at: Int[Tensor, " n"]
    end: Int[Tensor, " n"]
    length: int = RATE
    offset: int = 0

    @classmethod
    def stacked(cls, x: Float[Tensor, "n c s"]) -> "Windows":
        """Windows already cut, laid end to end so each one is its own word's window."""
        count, channels, samples = x.shape
        meg = x.transpose(0, 1).reshape(channels, count * samples)
        at = torch.arange(count, device=x.device) * samples
        return cls(meg, at, at + samples, samples)

    def __len__(self) -> int:
        return len(self.at)

    @property
    def device(self) -> torch.device:
        return self.meg.device

    def __call__(self, words: Int[Tensor, " k"], jitter: int = 0) -> Float[Tensor, "k c s"]:
        """The windows of `words`, in the signal's dtype; centring runs in float32 so float16 only rounds once.

        A window whose samples are all valid is centred by its plain mean (the rescale is then exactly 1). `jitter`
        moves each window's start by its own uniform draw in [-jitter, jitter] samples, still stopped at `end`."""
        start = self.at[words, None] + self.offset
        if jitter:
            start = (start + torch.randint_like(start, -jitter, jitter + 1)).clamp(min=0)
        position = start + torch.arange(self.length, device=self.meg.device)
        end = self.end[words, None]
        valid = (position < end)[:, None, :]
        cut = self.meg[:, torch.minimum(position, end - 1)].transpose(0, 1).float()
        cut = torch.where(valid, cut, 0.0)
        mean = cut.mean(dim=-1, keepdim=True) * (self.length / valid.sum(dim=-1, keepdim=True))
        return torch.where(valid, cut - mean, 0.0).to(self.meg.dtype)

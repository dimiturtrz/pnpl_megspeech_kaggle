"""Word windows cut on demand from one continuous device signal: a single gather, no per-word loop.

A word's window starts `offset` samples after its onset sample `at` and runs `length` samples, at the signal's
50 Hz `RATE` (seconds x `RATE`). It stops at the word's exclusive bound `end`: samples from `end` on are zero, so
a window longer than 1 s never reads past the stretch the Kaggle test would hold for that word. Each window then has
every channel's mean over its valid samples removed, or over the valid ones among its first `baseline`.
Training, validation and Kaggle test windows all come through here, from signals one `cache.run_cache.RunCache`
cached.
"""

from dataclasses import dataclass, replace
from functools import cached_property

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.constants import RATE


@dataclass(frozen=True)
class Windows:
    """Word `i`'s window is `meg[:, at[i] + offset : at[i] + offset + length]`, zero from `end[i]` on, centred on
    its valid samples (the first `baseline` of them when set)."""

    meg: Float[Tensor, "c t"]
    at: Int[Tensor, " n"]
    end: Int[Tensor, " n"]
    length: int = RATE
    offset: int = 0
    baseline: int = 0
    centred: bool = True
    """False reads windows as they are: features already computed from centred windows, not a signal."""
    begin: Int[Tensor, " n"] | None = None
    """Each word's first readable sample, its sentence epoch's start: samples before it are zero, as a Kaggle test epoch
    holds none before its first onset. None bounds only at the signal's start."""
    following: Int[Tensor, " n"] | None = None
    """Each word's next onset sample in its sentence (its `end` for the last): where `DataConfig.span_lag` cuts."""
    segment: int = 0
    """Samples of each span's continuous segment from its first onset, which `Split.batch` reads in place of word
    windows; 0 reads word windows."""

    @classmethod
    def stacked(cls, x: Float[Tensor, "n c s"], *, centred: bool = True) -> "Windows":
        """Windows already cut, laid end to end so each one is its own word's window."""
        count, channels, samples = x.shape
        meg = x.transpose(0, 1).reshape(channels, count * samples)
        at = torch.arange(count, device=x.device) * samples
        return cls(meg, at, at + samples, samples, centred=centred)

    def __len__(self) -> int:
        return len(self.at)

    def shifted(self, samples: int) -> "Windows":
        """Every window read `samples` later, its bounds kept."""
        return replace(self, at=self.at + samples)

    def unread(self) -> Int[Tensor, " m"]:
        """The words whose window holds no valid sample: it starts at or past the word's `end`."""
        return torch.nonzero(self.at + self.offset >= self.end).flatten()

    @property
    def device(self) -> torch.device:
        return self.meg.device

    def __call__(self, words: Int[Tensor, " k"], jitter: int = 0) -> Float[Tensor, "k c s"]:
        """The windows of `words`, in the signal's dtype; centring runs in float32 so float16 only rounds once, and
        saturates at the dtype's range: a sensor jump can centre past float16's, which no scaler clamp undoes once
        an augmentation's spread or spectrum turns the infinity into NaN.

        A window whose samples are all valid is centred by its plain mean (the rescale is then exactly 1). `jitter`
        moves each window's start by its own uniform draw in [-jitter, jitter] samples, still stopped at `end`."""
        start = self.at[words, None] + self.offset
        if jitter:
            start = (start + torch.randint_like(start, -jitter, jitter + 1)).clamp(min=0)
        position = start + torch.arange(self.length, device=self.meg.device)
        end = self.end[words, None]
        valid = ((position < end) & (position >= self.first(words)))[:, None, :]
        cut = self.meg[:, torch.minimum(position, end - 1).clamp(min=0)].transpose(0, 1).float()
        cut = torch.where(valid, cut, 0.0)
        counted = valid & (torch.arange(self.length, device=self.meg.device) < (self.baseline or self.length))
        mean = torch.where(counted, cut, 0.0).mean(dim=-1, keepdim=True)
        mean = mean * (self.length / counted.sum(dim=-1, keepdim=True).clamp(min=1)) * self.centred
        largest = torch.finfo(self.meg.dtype).max
        return torch.where(valid, cut - mean, 0.0).clamp(-largest, largest).to(self.meg.dtype)

    def first(self, words: Int[Tensor, " k"]) -> Int[Tensor, "k 1"]:
        """The first sample each of `words` may read."""
        return self.starts[words, None]

    @cached_property
    def starts(self) -> Int[Tensor, " n"]:
        """`begin`, or the signal's start for every word when it bounds none."""
        return self.at.new_zeros(len(self.at)) if self.begin is None else self.begin

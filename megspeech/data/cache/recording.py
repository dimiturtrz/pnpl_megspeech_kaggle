"""Cached runs end to end in one device tensor, each word's window a slice of it."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.windows import Windows
from megspeech.data.cache.run_cache import GB, RunCache
from megspeech.data.constants import RATE

STANDARDISE_SAMPLE = 20_000
"""Samples per channel a recording's quantiles are read from, strided across it."""


@dataclass(frozen=True)
class DeviceRecording:
    """Cached runs end to end in one device tensor: run `i` is `meg[:, offsets[i]:offsets[i + 1]]`.

    `words` is the word table restricted to the loaded runs, with `start` its run's first sample in `meg` and
    `at` the word's onset sample there.
    """

    meg: Float[Tensor, "c t"]
    offsets: Int[np.ndarray, " r1"]
    names: tuple[str, ...]
    words: pd.DataFrame

    @classmethod
    def load(
        cls,
        cache: RunCache,
        device: torch.device | str,
        names: Sequence[str] | None = None,
        *,
        standardised: bool = False,
    ) -> "DeviceRecording":
        """Load the named runs (default: every cached run) into one preallocated tensor, one run at a time, each
        `standardised` on its own if asked."""
        chosen = tuple(names) if names is not None else tuple(sorted(path.stem for path in cache.cache.glob("*.npy")))
        stored = [np.load(cache.path(name), mmap_mode="r") for name in chosen]
        offsets = np.r_[0, np.cumsum([signal.shape[1] for signal in stored])].astype(np.int64)
        meg = torch.empty((stored[0].shape[0], int(offsets[-1])), dtype=torch.float16, device=device)
        for begin, signal in zip(offsets, stored, strict=False):
            run = meg[:, begin : begin + signal.shape[1]]
            run.copy_(torch.from_numpy(np.array(signal)))
            if standardised:
                cls.standardise(run)
        words = cache.table(chosen)
        words["start"] = words["run"].map(dict(zip(chosen, offsets[:-1].tolist(), strict=True)))
        words["at"] = words["start"] + words["sample"]
        return cls(meg, offsets, chosen, words)

    @staticmethod
    @torch.no_grad()
    def standardise(run: Float[Tensor, "c s"]) -> None:
        """`run` in place as each channel's distance from its median in interquartile ranges: one recording's
        sensor gain and offset, which drift between sessions, taken out before the model's own scaler."""
        sample = run[:, :: max(1, run.shape[1] // STANDARDISE_SAMPLE)].float()
        levels = torch.tensor([0.25, 0.5, 0.75], device=run.device)
        low, median, high = torch.quantile(sample, levels, dim=1, keepdim=True)
        spread = (high - low).clamp(min=torch.finfo(torch.float32).tiny)
        run.copy_((run.float() - median) / spread)

    def windows(self, length: int = RATE) -> Windows:
        """Every word of `words`, in its order, as a `length`-sample window gathered from `meg`, ending at `stop` and
        reading nothing before its sentence's first onset."""
        return Windows(
            self.meg,
            self.samples(self.words["at"]),
            self.samples(self.words["start"] + self.words["stop"]),
            length,
            begin=self.samples(self.words["start"] + self.words["opening"]),
        )

    def samples(self, column: pd.Series) -> Int[Tensor, " n"]:
        """A word-table column of sample indices into `meg`, on its device."""
        return torch.from_numpy(column.to_numpy(dtype=np.int64, copy=True)).to(self.meg.device)

    @property
    def gigabytes(self) -> float:
        return self.meg.element_size() * self.meg.numel() / GB

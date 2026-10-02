"""Every subject-0 run as one continuous 50 Hz recording, cached in float16 and loaded whole onto the GPU.

The epoch cache fixes each window at a word onset; a continuous copy lets windows be cut on the fly at any
offset and length. The recordings are 250 Hz and already low-passed at 125 Hz; d'Ascoli 2025 decode at 50 Hz,
and 250 / 50 = 5 is an integer, so `resample_poly(x, 1, 5)` is a pure polyphase decimation whose FIR filter
removes everything above the new 25 Hz Nyquist before keeping every fifth sample — the same `DECIMATE` the
epoch windows use.

    uv run python -m megspeech.continuous     # cache every downloaded run under processed/continuous50
"""

import argparse
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from scipy.signal import resample_poly
from torch import Tensor

from core.paths import DataRoot
from megspeech.epochs import DECIMATE, HALF, PICO, SFREQ, WORKERS
from megspeech.runs import Run

LOG = logging.getLogger(__name__)

RATE = SFREQ // DECIMATE
"""50 Hz: the cached rate, and the rate a word onset in seconds is multiplied by to find its sample."""

DIRECTORY = f"continuous{RATE}"
WORDS = "words.tsv"
NO_SENTENCE = -1
GB = 1e9

MISSING = [""]
"""The only missing cell: with pandas' default NA spellings a heard `None` or `null` would read as no word."""


@dataclass(frozen=True)
class Decimator:
    """Writes each run's MEG as one `channels x time` float16 `.npy` in picotesla, plus one word-event table."""

    cache: Path
    decimate: int = DECIMATE

    def path(self, run: Run) -> Path:
        return self.cache / f"{run.name}.npy"

    def signal(self, run: Run) -> Float[np.ndarray, "c s"]:
        """The whole run decimated, in picotesla, saturated to the float16 range rather than overflowing to inf."""
        with h5py.File(run.meg) as stored:
            rate = float(stored.attrs["sample_frequency"])
            if rate != SFREQ:
                raise ValueError(f"{run.meg.name} is {rate} Hz, not {SFREQ}")
            meg = stored["data"][:]
        down = resample_poly(meg * np.float32(PICO), 1, self.decimate, axis=-1)
        return np.clip(down, -HALF, HALF).astype(np.float16)

    def write(self, run: Run) -> Path:
        """Cache one run; a run already cached is skipped, and a crash mid-write never leaves a final-named file."""
        path = self.path(run)
        if not path.exists():
            partial = path.with_suffix(".partial")
            with partial.open("wb") as handle:
                np.save(handle, self.signal(run))
            partial.replace(path)
        return path

    @staticmethod
    def words(run: Run) -> pd.DataFrame:
        """A run's heard words: onset in seconds into the run, the word as spoken, and its sentence in the run."""
        events = pd.read_csv(
            run.events,
            sep="\t",
            usecols=["kind", "segment", "sentenceidx", "timemeg"],
            keep_default_na=False,
            na_values=MISSING,
        )
        heard = events[(events["kind"] == "word") & events["segment"].notna()]
        return pd.DataFrame(
            {
                "run": run.name,
                "onset": heard["timemeg"].to_numpy(dtype=np.float64),
                "word": heard["segment"].to_numpy(),
                "sentence": heard["sentenceidx"].fillna(NO_SENTENCE).to_numpy(dtype=np.int64),
            }
        )

    def build(self, runs: Sequence[Run], workers: int = WORKERS) -> pd.DataFrame:
        """Cache every run in parallel, then write the word table of all of them."""
        self.cache.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(workers) as pool:
            for path in pool.map(self.write, runs):
                LOG.info("%s", path.name)
        table = pd.concat([self.words(run) for run in runs], ignore_index=True)
        table.to_csv(self.cache / WORDS, sep="\t", index=False)
        return table


@dataclass(frozen=True)
class Continuous:
    """Cached runs end to end in one device tensor: run `i` is `meg[:, offsets[i]:offsets[i + 1]]`.

    `words` is the word table restricted to the loaded runs, with `at` the word's onset sample in `meg`.
    """

    meg: Float[Tensor, "c t"]
    offsets: Int[np.ndarray, " r1"]
    names: tuple[str, ...]
    words: pd.DataFrame

    @classmethod
    def load(cls, cache: Path, device: torch.device | str, names: Sequence[str] | None = None) -> "Continuous":
        """Load the named runs (default: every cached run) into one preallocated tensor, one run at a time."""
        chosen = tuple(names) if names is not None else tuple(sorted(path.stem for path in cache.glob("*.npy")))
        stored = [np.load(cache / f"{name}.npy", mmap_mode="r") for name in chosen]
        offsets = np.r_[0, np.cumsum([signal.shape[1] for signal in stored])].astype(np.int64)
        meg = torch.empty((stored[0].shape[0], int(offsets[-1])), dtype=torch.float16, device=device)
        for begin, signal in zip(offsets, stored, strict=False):
            meg[:, begin : begin + signal.shape[1]] = torch.from_numpy(np.array(signal))
        start = dict(zip(chosen, offsets[:-1].tolist(), strict=True))
        table = pd.read_csv(cache / WORDS, sep="\t", keep_default_na=False, na_values=MISSING)
        words = table[table["run"].isin(start)].reset_index(drop=True)
        words["at"] = words["run"].map(start) + np.round(words["onset"] * RATE).astype(np.int64)
        return cls(meg, offsets, chosen, words)

    @property
    def gigabytes(self) -> float:
        return self.meg.element_size() * self.meg.numel() / GB


def main() -> None:
    """Cache every downloaded run at 50 Hz and report the cache's size and wall time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    decimator = Decimator(root.processed() / DIRECTORY)
    runs = Run.discover(root.raw())
    began = time.perf_counter()
    table = decimator.build(runs, args.workers)
    size = sum(decimator.path(run).stat().st_size for run in runs) / GB
    LOG.info(
        "%d runs, %d words, %.2f GB float16 in %s, %.0f s",
        len(runs),
        len(table),
        size,
        decimator.cache,
        time.perf_counter() - began,
    )


if __name__ == "__main__":
    main()

"""Every run of every subject as one continuous 50 Hz recording, cached in float16 and loaded whole onto the GPU.

A continuous copy lets word windows be gathered on the device at any offset and length (`windows.Windows`).
The recordings are 250 Hz and already low-passed at 125 Hz; d'Ascoli 2025 decode at 50 Hz, and 250 / 50 = 5
is an integer, so `resample_poly(x, 1, 5)` is a pure polyphase decimation whose FIR filter removes everything
above the new 25 Hz Nyquist before keeping every fifth sample. The holdout is cached the same way
(`megspeech.data.holdout`).

    uv run python -m megspeech.data.continuous                # cache every downloaded run under processed/continuous50
    uv run python -m megspeech.data.continuous --decimate 2   # the same at 125 Hz, under processed/continuous125
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
from jaxtyping import Bool, Float, Int
from scipy.signal import resample_poly
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.epochs import DECIMATE, HALF, PICO, RATE, SFREQ, WORKERS
from megspeech.data.runs import Run
from megspeech.data.windows import Windows

LOG = logging.getLogger(__name__)

DIRECTORY = "continuous{rate:g}"
"""A cache's directory under `processed/`, named by its rate; the default 50 Hz cache is `continuous50`."""
WORDS = "words.tsv"
NO_SENTENCE = -1
GB = 1e9

STANDARDISE_SAMPLE = 20_000
"""Samples per channel a recording's quantiles are read from, strided across it."""

TAIL = 1.0
"""Seconds a holdout sentence epoch runs past its sentence's last word (`sentence_extra_end_s`), measured from the
word's end: the epochs' tails after their last onset are never under 1.022 s, though one in nine training sentences
ends on a vocabulary word whose tail from its onset would be exactly 1 s."""

MISSING = [""]
"""The only missing cell: with pandas' default NA spellings a heard `None` or `null` would read as no word."""


@dataclass(frozen=True)
class Decimator:
    """Writes each run's MEG as one `channels x time` float16 `.npy` in picotesla, plus one word-event table."""

    cache: Path
    decimate: int = DECIMATE

    @classmethod
    def under(cls, processed: Path, decimate: int = DECIMATE) -> "Decimator":
        """The cache at `decimate` in its own directory of `processed`, so caches at different rates coexist."""
        return cls(processed / DIRECTORY.format(rate=SFREQ / decimate), decimate)

    @property
    def rate(self) -> float:
        """Samples per second the cache keeps."""
        return SFREQ / self.decimate

    def path(self, name: str) -> Path:
        return self.cache / f"{name}.npy"

    def ready(self) -> Path:
        """The cache directory, created if missing."""
        self.cache.mkdir(parents=True, exist_ok=True)
        return self.cache

    def signal(self, run: Run) -> Float[np.ndarray, "c s"]:
        """The whole run decimated, in picotesla, saturated to the float16 range rather than overflowing to inf."""
        with h5py.File(run.meg) as stored:
            rate = float(stored.attrs["sample_frequency"])
            if rate != SFREQ:
                raise ValueError(f"{run.meg.name} is {rate} Hz, not {SFREQ}")
            meg = stored["data"][:]
        return self.decimated(meg)

    def decimated(self, meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "c s"]:
        """One continuous 250 Hz stretch in tesla as cached: decimated, in picotesla, saturated, float16."""
        down = resample_poly(meg * np.float32(PICO), 1, self.decimate, axis=-1)
        return np.clip(down, -HALF, HALF).astype(np.float16)

    def length(self, samples: Int[np.ndarray, " n"]) -> Int[np.ndarray, " n"]:
        """Samples `decimated` keeps of `samples`: `resample_poly` rounds `samples / decimate` up."""
        return -(-samples // self.decimate)

    def write(self, run: Run) -> Path:
        """Cache one run; a run already cached is skipped."""
        path = self.path(run.name)
        if not path.exists():
            self.save(path, self.signal(run))
        return path

    @staticmethod
    def save(path: Path, signal: Float[np.ndarray, "c s"]) -> Path:
        """Write `signal` to `path` so that a crash mid-write never leaves a final-named file."""
        partial = path.with_suffix(".partial")
        with partial.open("wb") as handle:
            np.save(handle, signal)
        partial.replace(path)
        return path

    @staticmethod
    def words(run: Run) -> pd.DataFrame:
        """A run's heard words: onset in seconds into the run, the word as spoken, its sentence in the run, and the
        `end` its window stops at — `TAIL` after its sentence's last word ends, as a holdout sentence epoch does."""
        events = pd.read_csv(
            run.events,
            sep="\t",
            usecols=["kind", "segment", "sentenceidx", "timemeg", "duration"],
            keep_default_na=False,
            na_values=MISSING,
        )
        heard = events[(events["kind"] == "word") & events["segment"].notna()]
        sentence = heard["sentenceidx"].fillna(NO_SENTENCE).to_numpy(dtype=np.int64)
        finish = (heard["timemeg"] + heard["duration"].fillna(0.0)).groupby(sentence).transform("max")
        return pd.DataFrame(
            {
                "run": run.name,
                "onset": heard["timemeg"].to_numpy(dtype=np.float64),
                "word": heard["segment"].to_numpy(),
                "sentence": sentence,
                "end": finish.to_numpy(dtype=np.float64) + TAIL,
            }
        )

    def build(self, runs: Sequence[Run], workers: int = WORKERS) -> pd.DataFrame:
        """Cache every run in parallel, then write the word table of all of them."""
        self.ready()
        with ProcessPoolExecutor(workers) as pool:
            for path in pool.map(self.write, runs):
                LOG.info("%s", path.name)
        table = pd.concat([self.words(run) for run in runs], ignore_index=True)
        partial = self.cache / f"{WORDS}.partial"
        table.to_csv(partial, sep="\t", index=False)
        partial.replace(self.cache / WORDS)
        return table


@dataclass(frozen=True)
class Continuous:
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
        cache: Path,
        device: torch.device | str,
        names: Sequence[str] | None = None,
        *,
        standardised: bool = False,
        rate: float = RATE,
    ) -> "Continuous":
        """Load the named runs (default: every cached run) into one preallocated tensor, one run at a time, each
        `standardised` on its own if asked; `rate` is the cache's (`Decimator.rate`)."""
        chosen = tuple(names) if names is not None else tuple(sorted(path.stem for path in cache.glob("*.npy")))
        stored = [np.load(cache / f"{name}.npy", mmap_mode="r") for name in chosen]
        offsets = np.r_[0, np.cumsum([signal.shape[1] for signal in stored])].astype(np.int64)
        meg = torch.empty((stored[0].shape[0], int(offsets[-1])), dtype=torch.float16, device=device)
        for begin, signal in zip(offsets, stored, strict=False):
            run = meg[:, begin : begin + signal.shape[1]]
            run.copy_(torch.from_numpy(np.array(signal)))
            if standardised:
                cls.standardise(run)
        words = cls.table(cache, chosen, rate)
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
        run.copy_((run.float() - median) / (high - low))

    @staticmethod
    def table(cache: Path, names: Sequence[str], rate: float = RATE) -> pd.DataFrame:
        """The named runs' words, run by run in `names` order, with each word's onset `sample` in its run, the
        run's length in `samples` — read from the cache's headers, so no signal is loaded — and the exclusive
        `stop` sample of its `end`, never before its 1 s window nor past its run, all at the cache's `rate`."""
        samples = {name: np.load(cache / f"{name}.npy", mmap_mode="r").shape[1] for name in names}
        table = pd.read_csv(cache / WORDS, sep="	", keep_default_na=False, na_values=MISSING)
        table = table[table["run"].isin(samples)]
        order = table["run"].map({name: position for position, name in enumerate(names)}).to_numpy()
        table = table.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
        table["sample"] = np.round(table["onset"] * rate).astype(np.int64)
        table["opening"] = table.groupby(["run", "sentence"])["sample"].transform("min")
        table["samples"] = table["run"].map(samples).astype(np.int64)
        stop = np.round(table["end"].to_numpy() * rate).astype(np.int64)
        table["stop"] = np.minimum(np.maximum(stop, table["sample"] + round(rate)), table["samples"])
        return table

    @staticmethod
    def inside(table: pd.DataFrame, length: int = RATE, offset: int = 0) -> Bool[np.ndarray, " n"]:
        """Words whose window of `length` samples, `offset` after the onset, lies wholly within their run."""
        begin = table["sample"].to_numpy() + offset
        return (begin >= 0) & (begin + length <= table["samples"].to_numpy())

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


def main() -> None:
    """Cache every downloaded run at 250 Hz / `--decimate` and report the cache's size and wall time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--decimate", type=int, default=DECIMATE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    decimator = Decimator.under(root.processed(), args.decimate)
    runs = Run.discover(root.raw())
    began = time.perf_counter()
    table = decimator.build(runs, args.workers)
    size = sum(decimator.path(run.name).stat().st_size for run in runs) / GB
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

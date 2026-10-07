"""Every run of every subject as one continuous 50 Hz recording, cached in float16 for loading whole onto the GPU.

A continuous copy lets word windows be gathered on the device at any offset and length (`batching.windows.Windows`).
The recordings are 250 Hz and already low-passed at 125 Hz; d'Ascoli 2025 decode at 50 Hz, and 250 / 50 = 5
is an integer, so `resample_poly(x, 1, 5)` is a pure polyphase decimation whose FIR filter removes everything
above the new 25 Hz Nyquist before keeping every fifth sample. The Kaggle test is cached the same way
(`megspeech.data.cache.kaggle`).

    uv run python -m megspeech.data.cache.run_cache               # every downloaded run under processed/continuous50
    uv run python -m megspeech.data.cache.run_cache --decimate 2  # the same at 125 Hz, under processed/continuous125
    uv run python -m megspeech.data.cache.run_cache --cleaning ica  # ICA-cleaned, under processed/continuous50-ica
"""

import argparse
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from scipy.signal import resample_poly
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED, Geometry
from megspeech.data.cache.cleaning import Cleaner, Cleaning
from megspeech.data.constants import DECIMATE, HALF, MISSING, PICO, RATE, SFREQ, WORKERS
from megspeech.data.labels.vocabulary import ABSENT, IGNORE, PRIMARY, Vocabulary
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import KAGGLE_WORDS, Grouping
from megspeech.data.spans.spans import Spans

LOG = logging.getLogger(__name__)
FORMAT = "%(asctime)s %(message)s"

DIRECTORY = "continuous{rate:g}"
"""A cache's directory under `processed/`, named by its rate; the default 50 Hz cache is `continuous50`."""
WORD_TABLE = "words.tsv"
NO_SENTENCE = -1
GB = 1e9

TAIL = 1.0
"""Seconds a Kaggle test sentence epoch runs past its sentence's last word (`sentence_extra_end_s`), measured from the
word's end: the epochs' tails after their last onset are never under 1.022 s, though one in nine training sentences
ends on a vocabulary word whose tail from its onset would be exactly 1 s."""


@dataclass(frozen=True)
class RunCache:
    """Each run's MEG as one `channels x time` float16 `.npy` in picotesla under `cache`, plus one word-event table."""

    cache: Path
    decimate: int = DECIMATE
    cleaning: Cleaning = Cleaning.NONE

    @classmethod
    def under(cls, processed: Path, decimate: int = DECIMATE, cleaning: Cleaning = Cleaning.NONE) -> "RunCache":
        """The cache at `decimate` after `cleaning` in its own directory of `processed`, so caches at different rates
        and cleanings coexist."""
        name = "-".join(part for part in (DIRECTORY.format(rate=SFREQ / decimate), cleaning.value) if part)
        return cls(processed / name, decimate, cleaning)

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
        """The whole run cleaned and decimated, in picotesla, saturated to the float16 range rather than overflowing
        to inf."""
        with h5py.File(run.meg) as stored:
            rate = float(stored.attrs["sample_frequency"])
            if rate != SFREQ:
                raise ValueError(f"{run.meg.name} is {rate} Hz, not {SFREQ}")
            meg = Run.signal(stored)
        return self.decimated(self.cleaner(lambda: run.meg)(meg))

    def cleaner(self, layout: Callable[[], Path]) -> Cleaner:
        """The cache's `cleaning` of a recording laid out as the serialised run `layout` names."""
        return self.cleaning.cleaner(layout)

    def decimated(self, meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "c s"]:
        """One continuous 250 Hz stretch in tesla as cached: decimated, in picotesla, saturated, float16."""
        down = resample_poly(meg * np.float32(PICO), 1, self.decimate, axis=-1)
        return np.clip(down, -HALF, HALF).astype(np.float16)

    def length(self, samples: Int[np.ndarray, " n"]) -> Int[np.ndarray, " n"]:
        """Samples `decimated` keeps of `samples`: `resample_poly` rounds `samples / decimate` up."""
        return -(-samples // self.decimate)

    def write(self, name: str, produce: Callable[[], Float[np.ndarray, "c s"]]) -> Path:
        """Cache the signal `produce` makes as run `name`; a run already cached is skipped, never produced."""
        path = self.path(name)
        if not path.exists():
            self.save(path, produce())
        return path

    def cache_run(self, run: Run) -> Path:
        return self.write(run.name, partial(self.signal, run))

    @staticmethod
    @contextmanager
    def staged(path: Path) -> Iterator[Path]:
        """A sibling to write in place of `path`, moved onto it once written, so a crash mid-write never leaves a
        final-named, truncated file."""
        partial_path = path.with_name(f"{path.name}.partial")
        yield partial_path
        partial_path.replace(path)

    @classmethod
    def save(cls, path: Path, signal: Float[np.ndarray, "c s"]) -> Path:
        with cls.staged(path) as staging, staging.open("wb") as handle:
            np.save(handle, signal)
        return path

    @staticmethod
    def words(run: Run) -> pd.DataFrame:
        """A run's heard words: onset in seconds into the run, the word as spoken, its sentence in the run, and the
        `end` its window stops at — `TAIL` after its sentence's last word ends, as a Kaggle test sentence epoch does —
        and the `finish` of its own."""
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
                "finish": (heard["timemeg"] + heard["duration"].fillna(0.0)).to_numpy(dtype=np.float64),
            }
        )

    def build(self, runs: Sequence[Run], workers: int = WORKERS) -> pd.DataFrame:
        """Cache every run in parallel, then write the word table of all of them."""
        self.ready()
        logged = partial(logging.basicConfig, level=logging.INFO, format=FORMAT)
        with ProcessPoolExecutor(workers, initializer=logged) as pool:
            for path in pool.map(self.cache_run, runs):
                LOG.info("%s", path.name)
        return self.publish(pd.concat([self.words(run) for run in runs], ignore_index=True))

    def publish(self, table: pd.DataFrame) -> pd.DataFrame:
        """Write the cache's word table."""
        with self.staged(self.cache / WORD_TABLE) as staging:
            table.to_csv(staging, sep="\t", index=False)
        return table

    def published(self) -> pd.DataFrame:
        """The cache's word table as `publish` wrote it."""
        return pd.read_csv(self.cache / WORD_TABLE, sep="\t", keep_default_na=False, na_values=MISSING)

    def table(self, names: Sequence[str]) -> pd.DataFrame:
        """The named runs' words, run by run in `names` order, with each word's onset `sample` in its run, the
        run's length in `samples` — read from the cache's headers, so no signal is loaded — and the exclusive
        `stop` sample of its `end`, never before its 1 s window nor past its run, all at the cache's `rate`; and the
        same `opening`, `stop` and `following` under a `piece_` prefix for its piece of the Kaggle test's length,
        ending `TAIL` after the piece's last word ends."""
        samples = {name: np.load(self.path(name), mmap_mode="r").shape[1] for name in names}
        table = self.published()
        table = table[table["run"].isin(samples)]
        order = table["run"].map({name: position for position, name in enumerate(names)}).to_numpy()
        table = table.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
        table["sample"] = np.round(table["onset"] * self.rate).astype(np.int64)
        table["opening"] = table.groupby(["run", "sentence"])["sample"].transform("min")
        table["samples"] = table["run"].map(samples).astype(np.int64)
        self.bounded(table, "", "end", ["run", "sentence"])
        piece, _ = Spans.piece_codes(table.groupby(["run", "sentence"], sort=False).ngroup().to_numpy(), KAGGLE_WORDS)
        table["piece"] = piece
        finish = table["finish"] if "finish" in table else table["end"] - TAIL
        table["piece_end"] = finish.groupby(piece).transform("max") + TAIL
        self.bounded(table, "piece_", "piece_end", ["piece"])
        return table

    def bounded(self, table: pd.DataFrame, prefix: str, end: str, group: list[str]) -> None:
        """`table`'s `{prefix}opening`, `{prefix}stop` and `{prefix}following` samples: the first onset of each `group`,
        the `end` second there, never before a word's 1 s window nor past its run, and the next onset in the group."""
        table[f"{prefix}opening"] = table.groupby(group)["sample"].transform("min")
        stop = np.round(table[end].to_numpy() * self.rate).astype(np.int64)
        table[f"{prefix}stop"] = np.minimum(np.maximum(stop, table["sample"] + round(self.rate)), table["samples"])
        next_onset = table.groupby(group)["sample"].shift(-1)
        table[f"{prefix}following"] = next_onset.fillna(table[f"{prefix}stop"]).astype(np.int64)

    @staticmethod
    def carried(table: pd.DataFrame, grouping: Grouping, rate: float = RATE) -> pd.DataFrame:
        """The words of `table` (sampled at `rate`) that `grouping` carries and whose 1 s window fits in their run, with
        their `target` and `token` (the lower-cased word, hashed)."""
        labels = PRIMARY.labels(table["word"])
        target = np.where(labels == ABSENT, IGNORE, labels)
        return table.assign(target=target, token=Vocabulary.token(table["word"]))[
            grouping.keep(target) & RunCache.inside(table, round(rate))
        ]

    @staticmethod
    def inside(table: pd.DataFrame, length: int = RATE, offset: int = 0) -> Bool[np.ndarray, " n"]:
        """Words whose window of `length` samples, `offset` after the onset, lies wholly within their run."""
        begin = table["sample"].to_numpy() + offset
        return (begin >= 0) & (begin + length <= table["samples"].to_numpy())


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
        geometry: Geometry = AS_RECORDED,
    ) -> "DeviceRecording":
        """Load the named runs (default: every cached run) into one preallocated tensor, one run at a time, each in
        `geometry`: re-coloured if it applies, `standardised` on its own if asked, then projected if it projects."""
        chosen = tuple(names) if names is not None else tuple(sorted(path.stem for path in cache.cache.glob("*.npy")))
        stored = [np.load(cache.path(name), mmap_mode="r") for name in chosen]
        offsets = np.r_[0, np.cumsum([signal.shape[1] for signal in stored])].astype(np.int64)
        meg = torch.empty((geometry.rows(stored[0].shape[0]), int(offsets[-1])), dtype=torch.float16, device=device)
        for begin, signal, name in zip(offsets, stored, chosen, strict=False):
            run = torch.from_numpy(np.array(signal)).to(device)
            geometry.align(run, name)
            if standardised:
                cls.standardise(run)
            meg[:, begin : begin + signal.shape[1]].copy_(geometry.project(run, robust=standardised))
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
            following=self.samples(self.words["start"] + self.words["following"]),
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
    parser.add_argument("--cleaning", type=Cleaning, choices=list(Cleaning), default=Cleaning.NONE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    root = DataRoot.from_config()
    cache = RunCache.under(root.processed(), args.decimate, args.cleaning)
    runs = Run.everywhere(root)
    began = time.perf_counter()
    table = cache.build(runs, args.workers)
    size = sum(cache.path(run.name).stat().st_size for run in runs) / GB
    LOG.info(
        "%d runs, %d words, %.2f GB float16 in %s, %.0f s",
        len(runs),
        len(table),
        size,
        cache.cache,
        time.perf_counter() - began,
    )


if __name__ == "__main__":
    main()

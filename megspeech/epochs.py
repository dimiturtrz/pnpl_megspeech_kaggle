"""Cut every vocabulary word heard in a run into the window the holdout serves, and cache it.

The holdout gives each word 1 s of 250 Hz MEG from its onset, already band-passed 0.1–125 Hz. Training
windows are cut the same way and pass through the same `Window` transform the holdout will, so train and test
differ only in what was heard.

    uv run python -m megspeech.epochs        # cache every downloaded run under processed/
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from scipy.signal import resample_poly

from core.paths import DataRoot
from megspeech.runs import Run
from megspeech.vocabulary import ABSENT, PRIMARY, Vocabulary

LOG = logging.getLogger(__name__)

SFREQ = 250
WINDOW = SFREQ
"""1 s from onset — `LibriBrainCompetitionHoldout.WINDOW_SECONDS` at the recordings' 250 Hz."""

DECIMATE = 5
"""250 → 50 Hz: d'Ascoli 2025 decode at 50 Hz; `resample_poly`'s anti-alias filter keeps < 25 Hz."""

PICO = 1e12
"""Tesla → picotesla, so a float16 cache keeps the ~1e-13 T magnetometer signal out of subnormals."""

CHUNK = 512
WORKERS = 6


@dataclass(frozen=True)
class Window:
    """The one transform every 1 s window goes through, training and holdout alike."""

    decimate: int = DECIMATE

    def __call__(self, raw: Float[np.ndarray, "n c t"]) -> Float[np.ndarray, "n c s"]:
        """Decimate, express in picotesla, and remove each channel's mean over the window."""
        down = resample_poly(raw.astype(np.float64) * PICO, 1, self.decimate, axis=-1)
        return (down - down.mean(axis=-1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class Epochs:
    """Windows of heard vocabulary words with their labels and onsets (seconds into the run)."""

    x: Float[np.ndarray, "n c s"]
    y: Int[np.ndarray, " n"]
    onset: Float[np.ndarray, " n"]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=self.x.astype(np.float16), y=self.y, onset=self.onset)
        return path

    @classmethod
    def load(cls, path: Path) -> "Epochs":
        with np.load(path) as stored:
            return cls(stored["x"], stored["y"], stored["onset"])


@dataclass(frozen=True)
class Epocher:
    """Turns a run's continuous MEG into labelled windows, cached as one file per run."""

    cache: Path
    vocabulary: Vocabulary = PRIMARY
    window: Window = Window()

    def epoch(self, run: Run) -> Epochs:
        events = pd.read_csv(run.events, sep="\t", usecols=["kind", "segment", "timemeg"])
        words = events[(events["kind"] == "word") & events["segment"].notna()]
        labels = self.vocabulary.labels(words["segment"])
        onsets = words["timemeg"].to_numpy(dtype=np.float64)
        with h5py.File(run.meg) as stored:
            meg = stored["data"][:]
        starts = (onsets * SFREQ).astype(np.int64)
        keep = (labels != ABSENT) & (starts >= 0) & (starts + WINDOW <= meg.shape[1])
        starts = starts[keep]
        x = np.concatenate(
            [self.cut(meg, starts[at : at + CHUNK]) for at in range(0, len(starts), CHUNK)]
            or [np.empty((0, meg.shape[0], WINDOW // self.window.decimate), np.float32)]
        )
        return Epochs(x, labels[keep], onsets[keep])

    def cut(self, meg: Float[np.ndarray, "c t"], starts: Int[np.ndarray, " n"]) -> Float[np.ndarray, "n c s"]:
        return self.window(meg[:, starts[:, None] + np.arange(WINDOW)].transpose(1, 0, 2))

    def cached(self, run: Run) -> Epochs:
        path = self.cache / f"{run.name}.npz"
        return Epochs.load(path) if path.exists() else Epochs.load(self.epoch(run).save(path))

    def labelled(self, run: Run) -> tuple[Int[np.ndarray, " n"], Float[np.ndarray, " n"]]:
        """A cached run's labels and onsets, without reading its windows."""
        with np.load(self.cache / f"{run.name}.npz") as stored:
            return stored["y"], stored["onset"]


def main() -> None:
    """Cache the epochs of every downloaded run, a few runs at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    epocher = Epocher(root.processed() / "epochs")
    runs = Run.discover(root.raw())
    with ProcessPoolExecutor(WORKERS) as pool:
        for run, epochs in zip(runs, pool.map(epocher.cached, runs), strict=True):
            LOG.info("%s %s %d words", run.partition, run.name, len(epochs.y))


if __name__ == "__main__":
    main()

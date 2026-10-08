"""The Kaggle test as continuous 50 Hz segments, cached and gathered exactly as the training runs are.

Every Kaggle test epoch (`archive.Archive`) is a segment of continuous MEG, so each goes through `RunCache.decimated` on
its own — the transform a training run takes whole — and an archive's segments are cached end to end as one run of a
run cache. A Kaggle test word's window is then gathered by `batching.windows.Windows` from its onset sample, as a
training word's is. Only an isolated word, whose segment IS its window, still carries the decimation filter's edges:
the Kaggle test holds nothing either side of it.

Rows follow the competition's canonical order: subjects ascending, the sentence archive before the word archive,
epochs in stored order, a sentence's valid words in stored order.

    uv run python -m megspeech.data.cache.kaggle     # cache the Deep Kaggle test and report its windows
"""

import argparse
import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from pnpl.competition import write_submission
from pnpl.competition.holdout import DEEP_SUBJECTS, HOLDOUT_SUBDIR, TRACKS, LibriBrainCompetitionHoldout

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED, Geometry
from megspeech.data.cache.cleaning import UNCHANGED, Cleaner, Cleaning
from megspeech.data.cache.run_cache import DeviceRecording, RunCache
from megspeech.data.constants import DECIMATE, SFREQ
from megspeech.data.labels.vocabulary import SECONDARY
from megspeech.data.raw.runs import Run

LOG = logging.getLogger(__name__)

KAGGLE_DIR = "holdout"
"""The Kaggle test's cache, a directory inside the training cache that `DeviceRecording.load` never globs into; its
name predates the Kaggle naming, kept for the caches on disk."""


class ArchiveKind(StrEnum):
    """An archive's name after its subject, in canonical order."""

    SENTENCE = "holdout01_sentence"
    WORD = "holdout2_word"


@dataclass(frozen=True)
class Epochs:
    """Epoch `e` is the first `length[e]` samples of `meg[e]`; its words start `onset[e][mask[e]]` s into it."""

    length: Int[np.ndarray, " e"]
    onset: Float[np.ndarray, "e w"]
    mask: Bool[np.ndarray, "e w"]

    @classmethod
    def sentences(cls, archive: np.lib.npyio.NpzFile) -> "Epochs":
        return cls(
            archive["sentence_n_times"].astype(np.int64),
            archive["word_onsets_s"].astype(np.float64),
            archive["word_mask"].astype(bool),
        )

    @classmethod
    def words(cls, archive: np.lib.npyio.NpzFile) -> "Epochs":
        """One word per epoch, at its start; the epoch is the whole stored window."""
        count, _, samples = archive["meg"].shape
        return cls(np.full(count, samples, dtype=np.int64), np.zeros((count, 1)), np.ones((count, 1), dtype=bool))


READERS = {ArchiveKind.SENTENCE: Epochs.sentences, ArchiveKind.WORD: Epochs.words}


@dataclass(frozen=True)
class Archive:
    """One subject's Kaggle test archive of one kind, `subjXX_<kind>.npz` (laid out in `pnpl.competition.holdout`):
    250 Hz epochs, a sentence epoch one continuous stretch of up to nine words, a word epoch one isolated 1 s window."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    def epochs(self) -> Epochs:
        with np.load(self.path) as archive:
            rate = float(archive["sfreq"])
            if rate != SFREQ:
                raise ValueError(f"{self.path.name} is {rate} Hz, not {SFREQ}")
            return READERS[ArchiveKind(self.name.split("_", 1)[1])](archive)

    def signal(self, cache: RunCache, clean: Cleaner = UNCHANGED) -> Float[np.ndarray, "c s"]:
        """Every epoch decimated on its own, laid end to end, after the archive's epochs are `clean`ed as one
        recording at 250 Hz."""
        with np.load(self.path) as archive:
            meg = archive["meg"]
        length = self.epochs().length
        joined = clean(np.concatenate([epoch[:, :n] for epoch, n in zip(meg, length, strict=True)], axis=1))
        epochs = np.split(joined, np.cumsum(length)[:-1], axis=1)
        return np.concatenate([cache.decimated(epoch) for epoch in epochs], axis=1)

    def words(self, cache: RunCache) -> pd.DataFrame:
        """The archive's words in canonical order as a run cache's word table, each onset in seconds into the run
        and its `end` at its epoch's.

        A word's sample is its onset rounded within its own epoch, as a training word's is within its run; the
        onset is then stored on that sample, so offsetting by the epoch's start cannot tip a half-sample tie."""
        epochs = self.epochs()
        kept = cache.length(epochs.length)
        start = np.r_[0, np.cumsum(kept)[:-1]]
        epoch, slot = np.nonzero(epochs.mask)
        onset = epochs.onset[epoch, slot]
        rate = cache.rate
        begin = np.round(onset * rate)
        if ((begin < 0) | (begin + round(rate) > kept[epoch])).any():
            raise ValueError(f"{self.name}: a word's window runs past its epoch")
        at = (start[epoch] + begin) / rate
        end = (start + kept)[epoch] / rate
        return pd.DataFrame({"run": self.name, "onset": at, "word": "", "sentence": epoch, "end": end})


class Track(StrEnum):
    """A competition track: the Kaggle test subjects it scores, in canonical order."""

    DEEP = "deep"
    BROAD = "broad"
    """Subjects 1-39; 33-39 have no training data, so a model meets them zero-shot."""

    @property
    def subjects(self) -> tuple[int, ...]:
        return TRACKS[self.value]

    def kaggle(self, raw: Path) -> LibriBrainCompetitionHoldout:
        """The track's rows as the competition indexes them, from the archives under `raw`."""
        return LibriBrainCompetitionHoldout(self.value, data_path=raw, download=False)

    def write(self, raw: Path, primary: Float[np.ndarray, "n v"], path: Path) -> Path:
        """`primary` as the track's Kaggle submission, one row per test index; the secondary vocabulary uniform."""
        secondary = np.full((len(primary), len(SECONDARY.words)), 1 / len(SECONDARY.words))
        return write_submission(path, self.kaggle(raw).indices, primary, secondary)


@dataclass(frozen=True)
class KaggleCache:
    """The Kaggle test archives of `subjects`, cached into `cache` and gathered as one `Windows` in canonical order."""

    raw: Path
    cache: RunCache
    subjects: tuple[int, ...] = DEEP_SUBJECTS

    @classmethod
    def of(
        cls, root: DataRoot, decimate: int = DECIMATE, track: Track = Track.DEEP, cleaning: Cleaning = Cleaning.NONE
    ) -> "KaggleCache":
        """`track`'s Kaggle test cached inside the training cache at `decimate` after `cleaning`."""
        trained = RunCache.under(root.processed(), decimate, cleaning)
        kaggle = RunCache(trained.cache / KAGGLE_DIR, decimate, cleaning)
        return cls(root.raw() / HOLDOUT_SUBDIR, kaggle, track.subjects)

    def archives(self) -> list[Archive]:
        return [
            Archive(self.raw / f"subj{subject:02d}_{kind}.npz") for subject in self.subjects for kind in ArchiveKind
        ]

    def layout(self) -> Path:
        """A training run's h5, whose channel layout the archives share but do not record."""
        return Run.discover(self.raw.parent)[0].meg

    def build(self) -> pd.DataFrame:
        """Cache every archive, then write the word table of all of them."""
        self.cache.ready()
        archives = self.archives()
        clean = self.cache.cleaner(self.layout)
        for archive in archives:
            LOG.info("%s", self.cache.write(archive.name, partial(archive.signal, self.cache, clean)).name)
        return self.cache.publish(pd.concat([archive.words(self.cache) for archive in archives], ignore_index=True))

    def windows(
        self,
        device: torch.device | str,
        length: int,
        *,
        geometry: Geometry = AS_RECORDED,
    ) -> Windows:
        """Every Kaggle test row's `length`-sample window, in canonical order, from the built cache, each archive loaded
        in `geometry`."""
        names = [archive.name for archive in self.archives()]
        recording = DeviceRecording.load(self.cache, device, names, geometry=geometry)
        return recording.windows(length)


def main() -> None:
    """Cache a track's Kaggle test, then report its windows' shape, count against its rows, finite share, and the
    share of samples zeroed past their epoch's end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=float, default=1.0, help="window length in seconds")
    parser.add_argument("--decimate", type=int, default=DECIMATE)
    parser.add_argument("--track", type=Track, choices=list(Track), default=Track.DEEP)
    parser.add_argument("--cleaning", type=Cleaning, choices=list(Cleaning), default=Cleaning.NONE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    cache = KaggleCache.of(root, args.decimate, args.track, args.cleaning)
    cache.build()
    rows = len(args.track.kaggle(root.raw()))
    windows = cache.windows(args.device, round(args.window * cache.cache.rate))
    x = windows(torch.arange(len(windows), device=windows.device))
    finite = float(torch.isfinite(x).float().mean())
    zeroed = float((x == 0).all(dim=1).float().mean())
    shape = tuple(x.shape[1:])
    LOG.info(
        "%d windows of %s for %d Kaggle test rows, finite %.6f, zeroed %.6f", len(windows), shape, rows, finite, zeroed
    )


if __name__ == "__main__":
    main()

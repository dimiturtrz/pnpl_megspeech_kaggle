"""The Deep holdout as continuous 50 Hz segments, cached and gathered exactly as the training runs are.

Each holdout archive (`COMPETITION_HOLDOUT/subjXX_<source>.npz`, laid out in `pnpl.competition.holdout`) stores
epochs of 250 Hz MEG: a sentence epoch is one continuous stretch of `sentence_n_times` samples holding up to nine
words at `word_onsets_s`; a word epoch is one isolated 1 s window. Every epoch is a segment of continuous MEG, so
each goes through `continuous.Decimator.decimated` on its own — the transform a training run takes whole — and an
archive's segments are cached end to end as one run of a `continuous` cache. A holdout word's window is then
gathered by `windows.Windows` from its onset sample, as a training word's is. Only an isolated word, whose segment
IS its window, still carries the decimation filter's edges: the holdout holds nothing either side of it.

Rows follow the competition's canonical order: subjects ascending, the sentence archive before the word archive,
epochs in stored order, a sentence's valid words in stored order.

    uv run python -m megspeech.data.holdout     # cache the Deep holdout and report its windows
"""

import argparse
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from pnpl.competition.holdout import DEEP_SUBJECTS, HOLDOUT_SUBDIR, LibriBrainCompetitionHoldout

from core.paths import DataRoot
from megspeech.data.continuous import DIRECTORY, WORDS, Continuous, Decimator
from megspeech.data.epochs import RATE, SFREQ
from megspeech.data.windows import Windows

LOG = logging.getLogger(__name__)

HOLDOUT = "holdout"
"""The holdout's cache, a directory inside the training cache that `Continuous.load` never globs into."""

TRACK = "deep"


class Source(StrEnum):
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


READERS = {Source.SENTENCE: Epochs.sentences, Source.WORD: Epochs.words}


@dataclass(frozen=True)
class Archive:
    """One subject's holdout archive of one source."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    def epochs(self) -> Epochs:
        with np.load(self.path) as archive:
            rate = float(archive["sfreq"])
            if rate != SFREQ:
                raise ValueError(f"{self.path.name} is {rate} Hz, not {SFREQ}")
            return READERS[Source(self.name.split("_", 1)[1])](archive)

    def signal(self, decimator: Decimator) -> Float[np.ndarray, "c s"]:
        """Every epoch decimated on its own, laid end to end."""
        with np.load(self.path) as archive:
            meg = archive["meg"]
        length = self.epochs().length
        return np.concatenate([decimator.decimated(epoch[:, :n]) for epoch, n in zip(meg, length, strict=True)], axis=1)

    def write(self, decimator: Decimator) -> Path:
        """Cache the archive's signal as one run; an archive already cached is skipped."""
        path = decimator.path(self.name)
        if not path.exists():
            decimator.save(path, self.signal(decimator))
        return path

    def words(self, decimator: Decimator) -> pd.DataFrame:
        """The archive's words in canonical order as a `continuous` word table, each onset in seconds into the run.

        A word's sample is its onset rounded within its own epoch, as a training word's is within its run; the
        onset is then stored on that sample, so offsetting by the epoch's start cannot tip a half-sample tie."""
        epochs = self.epochs()
        kept = decimator.length(epochs.length)
        start = np.r_[0, np.cumsum(kept)[:-1]]
        epoch, slot = np.nonzero(epochs.mask)
        onset = epochs.onset[epoch, slot]
        begin = np.round(onset * RATE)
        if ((begin < 0) | (begin + RATE > kept[epoch])).any():
            raise ValueError(f"{self.name}: a word's window runs past its epoch")
        at = (start[epoch] + begin) / RATE
        return pd.DataFrame({"run": self.name, "onset": at, "word": "", "sentence": epoch})


@dataclass(frozen=True)
class HoldoutCache:
    """The holdout archives of `subjects`, cached by `decimator` and gathered as one `Windows` in canonical order."""

    raw: Path
    decimator: Decimator
    subjects: tuple[int, ...] = DEEP_SUBJECTS

    @classmethod
    def of(cls, root: DataRoot) -> "HoldoutCache":
        return cls(root.raw() / HOLDOUT_SUBDIR, Decimator(root.processed() / DIRECTORY / HOLDOUT))

    def archives(self) -> list[Archive]:
        return [Archive(self.raw / f"subj{subject:02d}_{source}.npz") for subject in self.subjects for source in Source]

    def build(self) -> pd.DataFrame:
        """Cache every archive, then write the word table of all of them."""
        cache = self.decimator.ready()
        archives = self.archives()
        for archive in archives:
            LOG.info("%s", archive.write(self.decimator).name)
        table = pd.concat([archive.words(self.decimator) for archive in archives], ignore_index=True)
        table.to_csv(cache / WORDS, sep="\t", index=False)
        return table

    def windows(self, device: torch.device | str) -> Windows:
        """Every holdout row's window, in canonical order, from the built cache."""
        return Continuous.load(self.decimator.cache, device, [archive.name for archive in self.archives()]).windows()


def main() -> None:
    """Cache the Deep holdout, then report its windows' shape, count against the holdout's rows, and finite share."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    cache = HoldoutCache.of(root)
    cache.build()
    rows = len(LibriBrainCompetitionHoldout(TRACK, data_path=root.raw(), download=False))
    windows = cache.windows(args.device)
    x = windows(torch.arange(len(windows), device=windows.device))
    finite = float(torch.isfinite(x).float().mean())
    LOG.info("%d windows of %s for %d holdout rows, finite %.6f", len(windows), tuple(x.shape[1:]), rows, finite)


if __name__ == "__main__":
    main()

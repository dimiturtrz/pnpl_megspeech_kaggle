"""One subject's holdout archive: its epochs, decimated end to end into one cached run, and its word table.

Each archive (`COMPETITION_HOLDOUT/subjXX_<kind>.npz`, laid out in `pnpl.competition.holdout`) stores epochs of
250 Hz MEG: a sentence epoch is one continuous stretch of `sentence_n_times` samples holding up to nine words at
`word_onsets_s`; a word epoch is one isolated 1 s window.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int

from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import SFREQ


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
    """One subject's holdout archive of one kind."""

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

    def signal(self, cache: RunCache) -> Float[np.ndarray, "c s"]:
        """Every epoch decimated on its own, laid end to end."""
        with np.load(self.path) as archive:
            meg = archive["meg"]
        length = self.epochs().length
        return np.concatenate([cache.decimated(epoch[:, :n]) for epoch, n in zip(meg, length, strict=True)], axis=1)

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

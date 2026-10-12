"""The speech envelope each training word's MEG window heard, at the window's own 50 Hz frames: the dense acoustic
target (Défossez et al. 2023 align MEG to speech continuously, not once per word).

Each trained Sherlock run's chapter (the audio and onsets `Heard` reads for the word-level speech targets, so the
alignment and its `Targets.aligned` mask are shared) is reduced to its log energy per 20 ms frame, standardised over
the chapter. A word's target is the `LENGTH` frames from its onset plus `OFFSET`, the span its MEG window covers, a
row of `<run>.npy` aligned with the run's word table (`Run.words`), NaN where there is none; `Speech.gather`
reads it as it reads the word-level targets. Validation and test runs' audio is never read, and a run whose dataset
ships no audio has no file, so no term.

uv run python -m megspeech.objective.targets.loudness
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.constants import RATE
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.speech import SILENCE, STRIDE, Chapter, Heard, Targets

NAME = "envelope"
"""The directory under processed/speech holding the envelope targets."""
LENGTH, OFFSET = DataConfig().cut()
"""Frames a word's window spans at 50 Hz and where it starts relative to the word's onset: the default geometry the
targets are cut for, which `RunConfig` holds an acoustic run to."""
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Loudness:
    """A chapter's log energy per `STRIDE`-sample frame, zero-mean and unit-variance over the chapter."""

    levels: Float[np.ndarray, " f"]

    @classmethod
    def of(cls, chapter: Chapter) -> "Loudness":
        frames = len(chapter.audio) // STRIDE
        energy = (chapter.audio[: frames * STRIDE].astype(np.float64).reshape(frames, STRIDE) ** 2).mean(axis=1)
        log = np.log(energy + SILENCE * energy.mean())
        return cls(((log - log.mean()) / log.std()).astype(np.float32))

    def windows(self, onset: Float[np.ndarray, " n"]) -> Targets:
        """Each word's `LENGTH` frames from its onset plus `OFFSET`; NaN past either end of the audio."""
        first = np.floor(np.nan_to_num(onset, nan=-1.0) * RATE).astype(np.int64) + OFFSET
        inside = np.isfinite(onset) & (first >= 0) & (first + LENGTH <= len(self.levels))
        frames = np.where(inside, first, 0)[:, None] + np.arange(LENGTH)
        return Targets(np.where(inside[:, None], self.levels[frames], np.nan).astype(np.float16))

    @staticmethod
    def directory(root: DataRoot) -> Path:
        return Targets.directory(root, NAME)

    @staticmethod
    def targets(entry: Heard) -> Targets:
        """`entry`'s words' envelope windows, those in a block whose audio is not the recording the events were timed
        on dropped."""
        chapter = Chapter.read(entry.chapters()[0])
        onset = entry.onsets()
        return Loudness.of(chapter).windows(onset).masked(Targets.aligned(chapter.contrast(onset)))

    @staticmethod
    def write(root: DataRoot) -> Path:
        directory = Loudness.directory(root)
        for entry in Heard.trained(Run.discover(root.raw())):
            path = directory / f"{entry.run.name}.npy"
            if path.exists():
                continue
            targets = Loudness.targets(entry)
            targets.save(path)
            LOG.info("%s: %d of %d words", entry.run.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Write every trained run's words' envelope windows under processed/speech/envelope; CPU only."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        LOG.info("wrote %s", Loudness.write(DataRoot.from_config()))


if __name__ == "__main__":
    Loudness.main()

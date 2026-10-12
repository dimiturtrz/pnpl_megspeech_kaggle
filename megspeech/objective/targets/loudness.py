"""The speech envelope each training word's MEG window heard, at the window's own 50 Hz frames: the dense acoustic
target (Défossez et al. 2023 align MEG to speech continuously, not once per word).

Each trained Sherlock run's chapter (the audio and onsets `Heard` reads for the word-level speech targets, so the
alignment and its `Targets.aligned` mask are shared) is reduced to its log energy per 20 ms frame, standardised over
the chapter. A word's target is the `LENGTH` frames from its onset plus `OFFSET`, the span its MEG window covers, a
row of `<run>.npy` aligned with the run's word table (`Run.words`), NaN where there is none; `Speech.gather`
reads it as it reads the word-level targets. Validation and test runs' audio is never read, and a run whose dataset
ships no audio has no file, so no term.

uv run python -m megspeech.objective.targets.loudness [--window 1.5 --delay 0]
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core.paths import DataRoot
from megspeech.data.config import Cut, DataConfig
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.speech import SAMPLING, SILENCE, STRIDE, Chapter, Heard, Targets

NAME = "envelope"
"""The directory under processed/speech holding the envelope targets."""
CUT = DataConfig().cut()
LENGTH, OFFSET = CUT
"""Frames a word's window spans at 50 Hz and where it starts relative to the word's onset, in the default geometry,
whose targets live under `NAME`; a run cut otherwise (`DataConfig.cut`) reads its own directory beside it."""
FRAMES = SAMPLING // STRIDE
"""Envelope frames per second, one per `STRIDE` audio samples: a run's windows are cut in frames of its MEG cache, so
only a cache at this rate (`DataConfig.rate`) reads its envelope frame for frame."""
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

    def windows(self, onset: Float[np.ndarray, " n"], cut: Cut) -> Targets:
        """Each word's `cut`: `length` frames from its onset plus `offset`; NaN past either end of
        the audio."""
        length, offset = cut
        first = np.floor(np.nan_to_num(onset, nan=-1.0) * FRAMES).astype(np.int64) + offset
        inside = np.isfinite(onset) & (first >= 0) & (first + length <= len(self.levels))
        frames = np.where(inside, first, 0)[:, None] + np.arange(length)
        return Targets(np.where(inside[:, None], self.levels[frames], np.nan).astype(np.float16))

    @staticmethod
    def directory(root: DataRoot, cut: Cut) -> Path:
        """`NAME` for the default geometry's targets, `NAME-<frames>-<offset>` for any other cut's."""
        return Targets.directory(root, NAME if cut == CUT else f"{NAME}-{cut.length}-{cut.offset}")

    @staticmethod
    def parsed(argv: list[str] | None = None) -> Cut:
        """The cut of the `--window` and `--delay` seconds in `argv`, each the default geometry's when absent."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--window", type=float, default=DataConfig.window)
        parser.add_argument("--delay", type=float, default=DataConfig.delay)
        known, _ = parser.parse_known_args(argv)
        return DataConfig(window=known.window, delay=known.delay).cut()

    @staticmethod
    def targets(entry: Heard, cut: Cut) -> Targets:
        """`entry`'s words' envelope windows, those in a block whose audio is not the recording the events were timed
        on dropped."""
        chapter = Chapter.read(entry.chapters()[0])
        onset = entry.onsets()
        return Loudness.of(chapter).windows(onset, cut).masked(Targets.aligned(chapter.contrast(onset)))

    @staticmethod
    def write(root: DataRoot, cut: Cut) -> Path:
        directory = Loudness.directory(root, cut)
        for entry in Heard.trained(Run.discover(root.raw())):
            path = directory / f"{entry.run.name}.npy"
            if path.exists():
                continue
            targets = Loudness.targets(entry, cut)
            targets.save(path)
            LOG.info("%s: %d of %d words", entry.run.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Write every trained run's words' envelope windows, cut as `--window` and `--delay` read; CPU only."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        LOG.info("wrote %s", Loudness.write(DataRoot.from_config(), Loudness.parsed()))


if __name__ == "__main__":
    Loudness.main()

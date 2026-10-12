"""The speech envelope each word's window heard in the non-English audiobook runs, MEG-SCANS' German parts
(`megspeech.data.raw.scans`) and SMN4Lang's Mandarin stories (`megspeech.data.raw.smn4lang`):
`megspeech.objective.targets.loudness`' targets for them, so the acoustic auxiliary (`--optim.acoustic`) trains on them.

A run's stimulus is one wav (each study names it by the run's key), its words' onsets the events' `timechapter`
(`Heard.onsets`), so the envelope, alignment mask and `<run>.npy` row layout are those of the LibriBrain runs.

uv run python -m megspeech.objective.targets.scansloudness [--window 1.5 --delay 0]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

from core.paths import DataRoot
from megspeech.data.config import Cut
from megspeech.data.raw.kymata import Study
from megspeech.data.raw.runs import Run
from megspeech.data.raw.scans import SCANS, STIMULI, TASK
from megspeech.data.raw.smn4lang import SMN4LANG, STORIES
from megspeech.objective.targets.loudness import Loudness
from megspeech.objective.targets.speech import Chapter, Heard, Targets

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Audiobook:
    """A MEG-SCANS run and the wav it played."""

    STUDY: ClassVar[Study] = SCANS

    run: Run

    def wav(self, root: DataRoot) -> Path:
        task, _, part = self.run.key
        book = task.removeprefix(TASK.format(book=""))
        return SCANS.home(root) / STIMULI / f"task-audiobook{book}_run-{part:02d}_stim.wav"

    def targets(self, root: DataRoot, cut: Cut) -> Targets:
        """The run's words' envelope windows, those in a block whose audio is not the recording dropped."""
        chapter = Chapter.read(self.wav(root))
        onset = Heard(self.run).onsets()
        return Loudness.of(chapter).windows(onset, cut).masked(Targets.aligned(chapter.contrast(onset)))

    @classmethod
    def write(cls, root: DataRoot, cut: Cut) -> Path:
        directory = Loudness.directory(root, cut)
        for run in Run.discover(root.processed(cls.STUDY.dataset)):
            path = directory / f"{run.name}.npy"
            if path.exists():
                continue
            targets = cls(run).targets(root, cut)
            targets.save(path)
            LOG.info("%s: %d of %d words", run.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Write every converted non-English run's words' envelope windows, cut as `--window` and `--delay` read; CPU
        only."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, cut = DataRoot.from_config(), Loudness.parsed()
        for kind in (Audiobook, Narrated):
            LOG.info("wrote %s", kind.write(root, cut))


@dataclass(frozen=True)
class Narrated(Audiobook):
    """An SMN4Lang run and the story it played."""

    STUDY: ClassVar[Study] = SMN4LANG

    @override
    def wav(self, root: DataRoot) -> Path:
        _, _, story = self.run.key
        return SMN4LANG.home(root) / STORIES / f"story_{story}.wav"


if __name__ == "__main__":
    Audiobook.main()

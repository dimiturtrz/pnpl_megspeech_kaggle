"""The speech envelope each MEG-SCANS word's window heard: `megspeech.objective.targets.loudness`' targets for the
German audiobook runs `megspeech.data.raw.scans` writes, so the acoustic auxiliary (`--optim.acoustic`) trains on them.

A run's stimulus is one wav (`Scans` names it by the run's book and part), its words' onsets the events' `timechapter`
(`Heard.onsets`), so the envelope, alignment mask and `<run>.npy` row layout are those of the LibriBrain runs.

uv run python -m megspeech.objective.targets.scansloudness
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.data.raw.scans import SCANS, STIMULI, TASK
from megspeech.objective.targets.loudness import Loudness
from megspeech.objective.targets.speech import Chapter, Heard, Targets

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Audiobook:
    """A MEG-SCANS run and the wav it played."""

    run: Run

    def wav(self, root: DataRoot) -> Path:
        task, _, part = self.run.key
        book = task.removeprefix(TASK.format(book=""))
        return SCANS.home(root) / STIMULI / f"task-audiobook{book}_run-{part:02d}_stim.wav"

    def targets(self, root: DataRoot) -> Targets:
        """The run's words' envelope windows, those in a block whose audio is not the recording dropped."""
        chapter = Chapter.read(self.wav(root))
        onset = Heard(self.run).onsets()
        return Loudness.of(chapter).windows(onset).masked(Targets.aligned(chapter.contrast(onset)))

    @staticmethod
    def write(root: DataRoot) -> Path:
        directory = Loudness.directory(root)
        for run in Run.discover(root.processed(SCANS.dataset)):
            path = directory / f"{run.name}.npy"
            if path.exists():
                continue
            targets = Audiobook(run).targets(root)
            targets.save(path)
            LOG.info("%s: %d of %d words", run.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Write every converted MEG-SCANS run's words' envelope windows under processed/speech/envelope; CPU only."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        LOG.info("wrote %s", Audiobook.write(DataRoot.from_config()))


if __name__ == "__main__":
    Audiobook.main()

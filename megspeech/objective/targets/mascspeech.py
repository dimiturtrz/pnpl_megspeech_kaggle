"""MEG-MASC's per-word speech targets: the wav2vec2 latents (`megspeech.objective.targets.speech`) and the envelope
windows (`megspeech.objective.targets.loudness`) of each converted run's heard words, read from the stimulus wav each
word was played in at its `start` there (`megspeech.data.raw.masc.Recording.heard`), so the speech, acoustic and audio
auxiliaries train on MASC as on LibriBrain.

A story plays as several wavs, so a run's rows are gathered wav by wav; every subject heard the same wavs, so each is
encoded once. `LAG` and the `Targets.aligned` mask are LibriBrain's: MASC's word-onset energy contrast peaks at
+0.06-0.08 s too, at a mean of ~1.6 against the 0.7 bar (6 recordings, 40 wavs).

uv run python -m megspeech.objective.targets.mascspeech [--window 1.5 --delay 0] [--speech]
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from core.paths import DataRoot
from megspeech.data.raw.events import Event, Events
from megspeech.data.raw.fetch import MASC
from megspeech.data.raw.masc import Masc, Recording
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.loudness import Loudness
from megspeech.objective.targets.speech import DEVICE, LAG, Chapter, Reading, Targets

LOG = logging.getLogger(__name__)

Cutter = Callable[[Float[np.ndarray, " n"]], Targets]
"""One wav's targets of the words at the given onsets in it."""


@dataclass(frozen=True)
class Told:
    """A converted MASC run and the recording it came from."""

    recording: Recording
    run: Run

    def targets(self, wavs: "Wavs") -> Targets:
        """The run's heard words' targets, in `Run.words` order, each wav's words cut from it by `wavs` and masked
        where their block of the wav is not the audio they were timed on."""
        heard = self.recording.heard()
        written = Events(self.run.events).words()[Event.SEGMENT].astype(str).tolist()
        if heard["word"].astype(str).tolist() != written:
            raise ValueError(f"{self.run.name}: its events' words are not its recording's heard words")
        parts = []
        for sound, group in heard.groupby("sound", sort=False):
            wav = self.recording.home / str(sound)
            onset = group["start"].to_numpy(dtype=np.float64) + LAG
            aligned = Targets.aligned(wavs.chapter(wav).contrast(onset))
            parts.append((group.index.to_numpy(), wavs(wav)(onset).masked(aligned)))
        vectors = np.full((len(heard), parts[0][1].vectors.shape[1]), np.nan, dtype=np.float16)
        for rows, targets in parts:
            vectors[rows] = targets.vectors
        return Targets(vectors)


@dataclass
class Wavs:
    """Each wav's targets maker, its chapter read and encoded once whichever subject heard it."""

    make: Callable[[Chapter], Cutter]
    chapters: dict[Path, Chapter] = field(default_factory=dict)
    made: dict[Path, Cutter] = field(default_factory=dict)

    def chapter(self, wav: Path) -> Chapter:
        if wav not in self.chapters:
            self.chapters[wav] = Chapter.read(wav)
        return self.chapters[wav]

    def __call__(self, wav: Path) -> Cutter:
        if wav not in self.made:
            self.made[wav] = self.make(self.chapter(wav))
        return self.made[wav]

    @staticmethod
    def envelope(cut: tuple[int, int]) -> "Wavs":
        return Wavs(lambda chapter: lambda onset: Loudness.of(chapter).windows(onset, cut))

    @staticmethod
    def speech(root: DataRoot, device: torch.device) -> "Wavs":
        reading = Reading()
        encoder = reading.load(root, device)
        return Wavs(lambda chapter: partial(Targets.pool, chapter.latents(encoder, device)))

    @staticmethod
    def write(root: DataRoot, directory: Path, wavs: "Wavs") -> Path:
        """Every converted MASC run's targets as `<run>.npy` under `directory`; a run already written is skipped."""
        runs = {run.name: run for run in Run.discover(root.processed(MASC.dataset))}
        for recording in Masc.under(root).recordings():
            path = directory / f"{recording.name}.npy"
            if recording.name not in runs or path.exists():
                continue
            targets = Told(recording, runs[recording.name]).targets(wavs)
            targets.save(path)
            LOG.info("%s: %d of %d words", recording.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Write every converted MASC run's envelope windows, cut as `--window` and `--delay` read, on the CPU, and with
        `--speech` its wav2vec2 targets on the GPU."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--speech", action="store_true")
        speech = parser.parse_known_args()[0].speech
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, cut = DataRoot.from_config(), Loudness.parsed()
        LOG.info("wrote %s", Wavs.write(root, Loudness.directory(root, cut), Wavs.envelope(cut)))
        if speech:
            LOG.info("wrote %s", Wavs.write(root, Targets.directory(root), Wavs.speech(root, torch.device(DEVICE))))


if __name__ == "__main__":
    Wavs.main()

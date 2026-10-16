"""MEG-SCANS (OpenNeuro ds006468, Gerrit et al., CC0) audiobook recordings rewritten as LibriBrain runs: each Triux
Neuromag-306 recording cleaned as Kymata's are (`megspeech.data.raw.sources.kymata_soto`: SSS with head-position
correction, mains notch, 0.1 Hz high-pass, 250 Hz), its German words in pnpl's columns.

The recordings are raw and carry the subject's own cross-talk and calibration files. A run is one audiobook part
(`task-audiobook{1,2}_run-0{1,2}`, ~9.5 min) played once; its forced-aligned words
(`stimuli/audiobooks/*_stim.TextGrid`, seconds from the wav's start) are placed on a clock fitted from the audio the
recording carries on `MISC001` (`kymata_soto_sync`): the 1 Hz pulse train the events file holds starts ~0.66 s before
the audio and is not used.

The words are German, so each segment carries the `GERMAN` prefix: `Vocabulary.spelled` keeps it, so none is spelled
as an English vocabulary word (`so`, `in`, `an`, `was` are both) and every one is `IGNORE` for the word and context
heads.
Only the words are written, no phonemes: the aligner's IPA is not the ARPAbet the phonetic target reads. The run's
`timechapter` column is the word's start in the stimulus wav, which `megspeech.objective.targets.scansloudness` reads
to cut the envelope target for the acoustic auxiliary.

    uv run python -m megspeech.data.raw.sources.meg_scans   # every audiobook recording, into processed/meg_scans
"""

import argparse
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import override

import pandas as pd

from core.paths import DataRoot
from megspeech.data.raw.events import Event, Events, Kind
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.serialised import Corpus, RunName
from megspeech.data.raw.sources.appleseed import Story
from megspeech.data.raw.sources.kymata_soto import Recording as Triux
from megspeech.data.raw.sources.kymata_soto import Study
from megspeech.data.raw.sources.kymata_soto_sync import Alignment, Clock
from megspeech.data.raw.sources.meg_masc import COLUMNS, Neuromag
from megspeech.data.raw.textgrid import TextGrid

LOG = logging.getLogger(__name__)

SUBJECT_OFFSET = 400
"""Added to a MEG-SCANS subject's number so no run shares a subject with LibriBrain's 0-32, MASC's 100+, Appleseed's
200+ or Kymata's 300+."""
SESSION = 1
TASK = "SCANSab{book}"
WORKERS = 4
"""Recordings converted at once: one holds ~3 GB in float64 while it is filtered, and SSS with movement doubles that."""
AUDIO_CHANNEL = "MISC001"
SOURCE = re.compile(r"sub-(?P<subject>\d+)_task-audiobook(?P<book>[12])_run-(?P<run>\d+)_meg\.fif")
STIMULI = Path("stimuli") / "audiobooks"
GERMAN = "de:"
"""Prefix of every segment: the words are German."""
COLUMNS_WITH_AUDIO = [*COLUMNS, Event.CHAPTER]
OFFSET_BOUNDS = (-0.2, 0.2)
"""Seconds from the coarse audio match a play's fitted start may lie: that match is biased by up to half the clip's
drift (~0.14 s), which the fit itself corrects."""

SCANS = Study("meg_scans")


@dataclass(frozen=True)
class Book:
    """One audiobook part's stimulus: its wav and forced alignment, named alike."""

    stem: Path

    @cached_property
    def alignment(self) -> Alignment:
        return Alignment.of(self.stem.with_name(f"{self.stem.name}.wav"))

    @cached_property
    def words(self) -> pd.DataFrame:
        """The spoken words in pnpl's columns, `timemeg` their start in the wav in seconds."""
        story = Story(TextGrid.read(self.stem.with_name(f"{self.stem.name}.TextGrid")))
        events = story.events(0.0)
        return events[events[Event.KIND] == Kind.WORD].reset_index(drop=True)


@dataclass(frozen=True)
class Recording(Triux):
    """One MEG-SCANS audiobook recording: a raw Triux `.fif` beside its events and sidecar."""

    @cached_property
    @override
    def parsed(self) -> dict[str, int]:
        parsed = Corpus.fields(SOURCE, self.fif.name)
        return {**parsed, "subject": parsed["subject"] + SUBJECT_OFFSET}

    @property
    @override
    def name(self) -> str:
        """The run name `Run` parses: subject past `SUBJECT_OFFSET`, one session, task `SCANSab<book>`, the part."""
        parsed = self.parsed
        return str(RunName(parsed["subject"], SESSION, TASK.format(book=parsed["book"]), parsed["run"]))

    @property
    @override
    def shared(self) -> dict[str, Path]:
        name = self.fif.name
        subject = name.split("_task")[0]
        return {
            "cross_talk": self.fif.with_name(f"{subject}_acq-crosstalk_meg.fif"),
            "calibration": self.fif.with_name(f"{subject}_acq-calibration_meg.dat"),
        }

    @property
    @override
    def audio_channel(self) -> str:
        return AUDIO_CHANNEL

    @property
    def book(self) -> Book:
        parsed = self.parsed
        return Book(self.dataset / STIMULI / f"task-audiobook{parsed['book']}_run-{parsed['run']:02d}_stim")

    @property
    def prefix(self) -> str:
        """What every segment starts with: the language its words are in."""
        return GERMAN

    @override
    def plays(self) -> list[Clock]:
        """The one play's clock in the saved MEG, fitted from the audio; a fit that is not believable is an error, as
        no constant model of this recording's drift exists."""
        alignment, recorded = self.book.alignment, self.audio()
        (nominal,) = alignment.locate(recorded, 1)
        clock = alignment.clock(recorded, nominal, OFFSET_BOUNDS)
        if not clock.fitted:
            raise ValueError(f"{self.fif.name}: the audio fit is not believable")
        return [clock]

    @override
    def events_at(self, plays: list[Clock]) -> pd.DataFrame:
        """The words on the clock of `plays`' one play, `timechapter` their start in the wav."""
        (clock,) = plays
        words = self.book.words
        placed = clock.place(words).assign(segment=self.prefix + words[Event.SEGMENT], timechapter=words[Event.ONSET])
        return Events.ordered(placed, COLUMNS_WITH_AUDIO)


@dataclass(frozen=True)
class Scans(Corpus[Neuromag]):
    """MEG-SCANS' recordings and the LibriBrain-shaped runs they become under `processed/meg_scans`."""

    @classmethod
    def under(cls, root: DataRoot) -> "Scans":
        return cls(SCANS.home(root), root.processed(SCANS.dataset))

    def recordings(self, subjects: tuple[int, ...] = ()) -> Iterator[Recording]:
        """Every audiobook recording of `subjects` (all if none; numbers as `Recording.parsed` has them) that is
        non-empty and beside its events."""
        for fif in sorted(self.source.glob("sub-*/meg/sub-*_task-audiobook*_meg.fif")):
            if SOURCE.fullmatch(fif.name) is None:
                continue
            recording = Recording(fif)
            chosen = not subjects or recording.parsed["subject"] in subjects
            if chosen and fif.stat().st_size and recording.sidecar("events.tsv").exists():
                yield recording


def main() -> None:
    """Convert every audiobook recording not yet converted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--subjects", type=int, nargs="*", default=[], help="run subject numbers (401+); default all")
    parser.add_argument("--target", type=Path, default=None, help="write the runs here instead of processed/meg_scans")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    scans, layout = Scans.under(root), Neuromag.of(NEUROMAG.home(root))
    if args.target is not None:
        scans = Scans(scans.source, args.target)
    recordings = list(scans.recordings(tuple(args.subjects)))
    with ProcessPoolExecutor(args.workers) as pool:
        written = list(pool.map(scans.convert, recordings, [layout] * len(recordings)))
    LOG.info("%d MEG-SCANS runs under %s", len(written), scans.target)


if __name__ == "__main__":
    main()

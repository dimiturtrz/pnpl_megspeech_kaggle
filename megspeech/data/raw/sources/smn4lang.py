"""SMN4Lang (OpenNeuro ds004078, Wang et al., CC0) story recordings rewritten as LibriBrain runs: each Triux
Neuromag-306 recording mains-notched, 0.1 Hz high-passed and resampled to 250 Hz, its Mandarin words in pnpl's columns.

The fif files on OpenNeuro are not raw: MaxFilter 2.2.15 already applied tSSS with the site's cross-talk and
fine-calibration files (`proc_history`), and MNE refuses SSS twice, so none is redone and the head stays where the
site's MaxFilter put it. The sidecars record 60 Hz mains, but the recordings carry their line at 50 Hz (a magnetometer
spectrum peaks ten-fold there and not at 60 Hz), so `MAINS` is notched.

A run is one of 60 stories (`run-N` plays `stimuli/audio/story_N.wav`, 4-7 min) heard once. The recording carries no
audio channel, so its clock is the stimulus-start trigger (`Beg` in the events file, seconds from the saved file's first
sample); in two runs (sub-08 run-16, sub-09 run-7) the logged start is not the audio's, and the story's end trigger less
its duration is used instead, as the dataset's README prescribes. Words are the dataset's forced alignment
(`derivatives/annotations/time_align/word-level`), whose times are shifted `ALIGNMENT_SHIFT` to the fMRI clock and
whose punctuation (zero-length entries) splits sentences and is then dropped.

The words are Mandarin, so each segment carries the `MANDARIN` prefix and is `IGNORE` for the word and context heads;
the run's `timechapter` (the word's start in the wav) feeds `megspeech.objective.targets.scansloudness`' acoustic
targets, as MEG-SCANS' does.

    uv run python -m megspeech.data.raw.sources.smn4lang                # every story recording, into processed/smn4lang
"""

import argparse
import logging
import re
import unicodedata
import wave
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import override

import mne
import pandas as pd
from scipy.io import loadmat

from core.paths import DataRoot
from megspeech.data.constants import QUIET, SFREQ
from megspeech.data.raw.events import Event, Kind
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.mains import Mains
from megspeech.data.raw.serialised import Corpus, RunName
from megspeech.data.raw.sources.appleseed import Story
from megspeech.data.raw.sources.kymata import Study
from megspeech.data.raw.sources.kymatasync import Clock
from megspeech.data.raw.sources.masc import HIGHPASS, Neuromag
from megspeech.data.raw.sources.scans import Book
from megspeech.data.raw.sources.scans import Recording as Played
from megspeech.data.raw.textgrid import Interval, TextGrid

LOG = logging.getLogger(__name__)

SUBJECT_OFFSET = 500
"""Added to an SMN4Lang subject's number so no run shares a subject with LibriBrain's 0-32, MASC's 100+, Appleseed's
200+, Kymata's 300+ or MEG-SCANS' 400+."""
SESSION = 1
TASK = "SMN4Lang"
WORKERS = 4
SOURCE = re.compile(r"sub-(?P<subject>\d+)_task-RDR_run-(?P<run>\d+)_meg\.fif")
STORIES = Path("stimuli") / "audio"
WORD_TIMES = Path("derivatives") / "annotations" / "time_align" / "word-level"
ALIGNMENT_SHIFT = 10.65
"""Seconds the alignment's times were moved to meet the fMRI clock (8 s blank and 2.65 s of instruction before each
story); taken off, they count from the wav's start."""
MANDARIN = "zh:"
"""Prefix of every segment: the words are Mandarin."""
MAINS = Mains(50.0)
START, END = "Beg", "End"
FULL_STOPS = frozenset("。！？!?")
"""The punctuation that ends a sentence; the alignment spells it as a zero-length word."""
SPAN_TOLERANCE = 1.0
"""Seconds the start-to-end trigger span may miss the story's duration before the start is not trusted: good runs
miss by -45 to +28 ms, the two broken ones by 6-7.5 s."""


@dataclass(frozen=True)
class Punctuated(Story):
    """A story whose alignment spells its punctuation as words: a sentence ends at a full stop, not at a pause (a
    Mandarin narrator pauses inside sentences as often as between them)."""

    @cached_property
    @override
    def words(self) -> pd.DataFrame:
        """The spoken words, punctuation dropped, each flagged `new` if it starts a sentence."""
        tier = self.grid.words
        sentence = tier[Interval.TEXT].isin(FULL_STOPS).cumsum().shift(fill_value=0)
        spoken = ~tier[Interval.TEXT].map(Punctuated.mark)
        words = tier[spoken].reset_index(drop=True)
        words["sentence"] = pd.factorize(sentence[spoken])[0]
        words["new"] = words["sentence"].diff().ne(0)
        words[Event.WORD] = words.groupby("sentence").cumcount()
        return words

    @staticmethod
    def mark(text: str) -> bool:
        """Whether `text` is no word: empty, or punctuation only."""
        return all(unicodedata.category(character).startswith("P") for character in text)


@dataclass(frozen=True)
class Narration(Book):
    """One story's stimulus: its wav and the dataset's word alignment."""

    @property
    def dataset(self) -> Path:
        return self.stem.parents[2]

    @cached_property
    @override
    def words(self) -> pd.DataFrame:
        """The spoken words in pnpl's columns, `timemeg` their start in the wav in seconds."""
        aligned = loadmat(self.dataset / WORD_TIMES / f"{self.stem.name}_word_time.mat")
        tier = pd.DataFrame(
            {
                Interval.START: aligned["start"].ravel() - ALIGNMENT_SHIFT,
                Interval.END: aligned["end"].ravel() - ALIGNMENT_SHIFT,
                Interval.TEXT: [str(word).strip() for word in aligned["word"].ravel()],
            }
        )
        events = Punctuated(TextGrid.aligned(tier, tier.iloc[:0])).events(0.0)
        return events[events[Event.KIND] == Kind.WORD].reset_index(drop=True)

    @cached_property
    def duration(self) -> float:
        """The wav's length in seconds."""
        with wave.open(str(self.stem.with_name(f"{self.stem.name}.wav"))) as audio:
            return audio.getnframes() / audio.getframerate()


@dataclass(frozen=True)
class Recording(Played):
    """One SMN4Lang story recording: a MaxFiltered Triux `.fif` beside its events and sidecar."""

    @cached_property
    @override
    def parsed(self) -> dict[str, int]:
        fields = Corpus.fields(SOURCE, self.fif.name)
        return {**fields, "subject": fields["subject"] + SUBJECT_OFFSET}

    @property
    @override
    def name(self) -> str:
        """The run name `Run` parses: subject past `SUBJECT_OFFSET`, one session, task `SMN4Lang`, the story."""
        parsed = self.parsed
        return str(RunName(parsed["subject"], SESSION, TASK, parsed["run"]))

    @property
    @override
    def book(self) -> Narration:
        return Narration(self.dataset / STORIES / f"story_{self.parsed['run']}")

    @property
    @override
    def prefix(self) -> str:
        return MANDARIN

    @property
    @override
    def mains(self) -> Mains:
        return MAINS

    @override
    def cleaned(self) -> mne.io.BaseRaw:
        """The recording's MEG channels as the site's MaxFilter left them (tSSS), mains notched, high-passed, 250 Hz."""
        raw = mne.io.read_raw_fif(self.fif, preload=True, verbose=QUIET)
        raw.pick("meg")
        self.mains.notch(raw)
        raw.filter(HIGHPASS, None, verbose=QUIET)
        return raw.resample(SFREQ, verbose=QUIET)

    def marks(self) -> dict[str, float]:
        """The story's start and end triggers, in seconds from the saved file's first sample."""
        table = pd.read_csv(self.sidecar("events.tsv"), sep="\t")
        return table.groupby("trial_type")["onset"].first().astype(float).to_dict()

    @override
    def plays(self) -> list[Clock]:
        """The one play's clock: the start trigger, or the end trigger less the story when the two disagree."""
        marks, duration = self.marks(), self.book.duration
        trusted = abs(marks[END] - marks[START] - duration) <= SPAN_TOLERANCE
        return [Clock(marks[START] if trusted else marks[END] - duration, 0.0, 0.0)]


SMN4LANG = Study("smn4lang", "smn4lang")


@dataclass(frozen=True)
class Smn4Lang(Corpus[Neuromag]):
    """SMN4Lang's recordings and the LibriBrain-shaped runs they become under `processed/smn4lang`."""

    @classmethod
    def under(cls, root: DataRoot) -> "Smn4Lang":
        return cls(SMN4LANG.home(root), root.processed(SMN4LANG.dataset))

    def recordings(self, subjects: tuple[int, ...] = ()) -> Iterator[Recording]:
        """Every story recording of `subjects` (all if none; numbers as `Recording.parsed` has them) that is
        non-empty and beside its events."""
        for fif in sorted(self.source.glob("sub-*/meg/sub-*_task-RDR_run-*_meg.fif")):
            recording = Recording(fif)
            chosen = not subjects or recording.parsed["subject"] in subjects
            if chosen and fif.stat().st_size and recording.sidecar("events.tsv").exists():
                yield recording


def main() -> None:
    """Convert every story recording not yet converted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--subjects", type=int, nargs="*", default=[], help="run subject numbers (501+); default all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    corpus, layout = Smn4Lang.under(root), Neuromag.of(NEUROMAG.home(root))
    recordings = list(corpus.recordings(tuple(args.subjects)))
    with ProcessPoolExecutor(args.workers) as pool:
        written = list(pool.map(corpus.convert, recordings, [layout] * len(recordings)))
    LOG.info("%d SMN4Lang runs under %s", len(written), corpus.target)


if __name__ == "__main__":
    main()

"""Kymata-SOTO (Thwaites and Yang; OSF qdzgh, licensed CC-BY 4.0 there, though its BIDS description says CC0)
rewritten as LibriBrain runs: each Triux Neuromag-306 recording Maxwell-filtered the way LibriBrain's are, at 250 Hz,
its word and phoneme events in pnpl's columns.

The recordings are raw (no SSS applied) and carry continuous head localisation, so the cleaning is LibriBrain's in its
order: bad channels, SSS with the cross-talk and fine-calibration files and head-position correction onto one standard
head position, mains notch, 0.1 Hz high-pass, 250 Hz. Both arrays are Neuromag-306, so no field mapping is needed:
the channels are only put in LibriBrain's order.

Each run plays the same ~400 s podcast clip twice; the clip's forced alignment (`stimuli/audio/mfa_*.txt`, seconds
from the clip's start) is placed on each play's fitted clock (`kymatasync`): the audio the recording carries on
`MISC006` gives the play's start and the stimulus clock's drift, the start trigger only says where to look.

    uv run python -m megspeech.data.raw.sources.kymata                  # every recording, into processed/kymata
    uv run python -m megspeech.data.raw.sources.kymata --events-only  # rewrite events of converted runs, MEG untouched
"""

import argparse
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from jaxtyping import Float
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter

from core.paths import DataRoot
from megspeech.data.constants import QUIET, SFREQ
from megspeech.data.raw.events import Event, Events
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.mains import Mains
from megspeech.data.raw.serialised import Corpus, RunName
from megspeech.data.raw.sources.appleseed import Story
from megspeech.data.raw.sources.kymatasync import Alignment, Clock, Envelope
from megspeech.data.raw.sources.masc import COLUMNS, HIGHPASS, ORIGIN, Neuromag
from megspeech.data.raw.textgrid import Interval, TextGrid

LOG = logging.getLogger(__name__)

SUBJECT_OFFSET = 300
"""Added to a Kymata subject's number so no run shares a subject with LibriBrain's 0-32,
MASC's 100+ or Appleseed's 200+."""
RESCAN_OFFSET = 20
"""Added again to a `b` subject (`E3b`, a second recruitment under the number of `E3`) so it is a subject of its own."""
SESSION = 1
TASK = "Kymata"
WORKERS = 4
"""Recordings converted at once: one holds ~3 GB in float64 while it is filtered, and SSS with movement doubles that."""
EVENT_WORKERS = 8
"""Events-only rewrites at once: they read one audio channel, not the whole recording."""
AUDIO_CHANNEL = "MISC006"
"""The channel carrying the stimulus audio."""
SOURCE = re.compile(r"sub-E(?P<subject>\d+)(?P<rescan>b?)_task-en_run-(?P<run>\d+)_meg\.fif")
STIMULUS_TRIGGER = 3
"""The trigger code that marks the start of each of a run's two plays of the clip."""
CROSS_TALK = "ct_sparse_triux.fif"
CALIBRATION = "sss_cal_triux.dat"
POSE_STEP = 0.1
"""Seconds between head-pose fits (MNE's default is 0.01 s: ten times the cost of the 6 min per recording it already
takes, for a head that moves over seconds)."""
ALIGNMENT = Path("stimuli") / "audio"
STIMULUS = "stimulus.wav"
SILENCES = {"<sp>", "spn"}
"""The aligner's pause and spoken-noise labels: neither is a word nor a phone."""


@dataclass(frozen=True)
class Study:
    """A dataset kept under `raw/<raw>` as downloaded by hand and rewritten under `processed/<dataset>`."""

    raw: str
    dataset: str

    def home(self, root: DataRoot) -> Path:
        return root.raw(self.raw)


KYMATA = Study("kymata_soto", "kymata")


@dataclass(frozen=True)
class Clip:
    """The played clip's forced alignment: items and their start times, each lasting until the next one starts."""

    directory: Path

    def tier(self, kind: str) -> pd.DataFrame:
        """`kind`'s (`word` or `phoneme`) items as a TextGrid-style tier with silences blanked."""
        text = (self.directory / f"mfa_{kind}.txt").read_text(encoding="utf-8").split("\n")
        start = np.loadtxt(self.directory / f"mfa_{kind}_stime.txt")
        items = [label for label in text if label]
        end = np.append(start[1:], start[-1])
        blank = [label in SILENCES for label in items]
        return pd.DataFrame({Interval.START: start, Interval.END: end, Interval.TEXT: np.where(blank, "", items)})

    @cached_property
    def alignment(self) -> Alignment:
        return Alignment.of(self.directory / STIMULUS)

    @cached_property
    def story(self) -> Story:
        return Story(TextGrid.aligned(self.tier("word"), self.tier("phoneme")))


@dataclass(frozen=True)
class Recording:
    """One Kymata BIDS recording: a raw Triux `.fif` beside its events and sidecar."""

    fif: Path

    @cached_property
    def parsed(self) -> dict[str, int]:
        match = Corpus.matched(SOURCE, self.fif.name)
        subject = int(match["subject"]) + SUBJECT_OFFSET + (RESCAN_OFFSET if match["rescan"] else 0)
        return {"subject": subject, "run": int(match["run"])}

    @property
    def name(self) -> str:
        """The run name `Run` parses: subject past `SUBJECT_OFFSET`, one session, task `Kymata`, the recording's run."""
        parsed = self.parsed
        return str(RunName(parsed["subject"], SESSION, TASK, parsed["run"]))

    def sidecar(self, suffix: str) -> Path:
        """The file beside the recording ending `suffix`."""
        fif = self.fif
        return fif.with_name(f"{fif.stem.removesuffix('meg')}{suffix}")

    @property
    def dataset(self) -> Path:
        """The dataset's root, two levels above the file."""
        return self.fif.parents[2]

    @property
    def shared(self) -> dict[str, Path]:
        """The cross-talk and fine-calibration files SSS reads."""
        return {"cross_talk": self.dataset / CROSS_TALK, "calibration": self.dataset / CALIBRATION}

    @property
    def audio_channel(self) -> str:
        return AUDIO_CHANNEL

    def head_positions(self, raw: mne.io.BaseRaw) -> Float[np.ndarray, "n 10"]:
        """The head's pose over the recording, fitted from its head-position indicator coils."""
        amplitudes = mne.chpi.compute_chpi_amplitudes(raw, t_step_min=POSE_STEP, verbose=QUIET)
        locations = mne.chpi.compute_chpi_locs(raw.info, amplitudes, verbose=QUIET)
        return mne.chpi.compute_head_pos(raw.info, locations, verbose=QUIET)

    @property
    def mains(self) -> Mains:
        return Mains.of(self.sidecar("meg.json"))

    def cleaned(self) -> mne.io.BaseRaw:
        """The recording's MEG channels: SSS with head-position correction, mains notched, high-passed, 250 Hz."""
        raw = mne.io.read_raw_fif(self.fif, preload=True, verbose=QUIET)
        shared = self.shared
        positions = self.head_positions(raw)
        found = find_bad_channels_maxwell(raw, verbose=QUIET, **shared)
        raw.info["bads"] = sorted({*raw.info["bads"], *found[0], *found[1]})
        raw = maxwell_filter(raw, head_pos=positions, destination=ORIGIN, verbose=QUIET, **shared)
        raw.pick("meg")
        self.mains.notch(raw)
        raw.filter(HIGHPASS, None, verbose=QUIET)
        return raw.resample(SFREQ, verbose=QUIET)

    def written(self, target: Neuromag) -> tuple[Float[np.ndarray, "c t"], dict[str, object]]:
        """The recording's field at `target`'s channels in tesla, float32, with its cleaning's record as attributes."""
        raw = self.cleaned()
        names = [name.replace(" ", "") for name in raw.ch_names]
        order = [names.index(name) for name in target.names]
        bads = ", ".join(raw.info["bads"])
        LOG.info("%s: bads %s", self.fif.name, bads)
        attrs = {**target.attrs, "bad_channels": bads, "mains": self.mains.frequency}
        return np.asarray(raw.get_data(picks=np.array(order)), dtype=np.float32), attrs

    @property
    def clip(self) -> Clip:
        return Clip(self.dataset / ALIGNMENT)

    def triggers(self) -> list[float]:
        """The seconds into the saved MEG of each play's start trigger: the events' onsets count from the
        acquisition's start, the file's first sample comes `first_time` after it."""
        table = pd.read_csv(self.sidecar("events.tsv"), sep="	")
        first = mne.io.read_raw_fif(self.fif, preload=False, verbose=QUIET).first_time
        return (table.loc[table["trial_type"] == STIMULUS_TRIGGER, "onset"].astype(float) - first).tolist()

    def audio(self) -> Envelope:
        """The envelope of the stimulus audio the recording carries."""
        raw = mne.io.read_raw_fif(self.fif, preload=False, verbose=QUIET)
        return Envelope.of(raw.get_data(picks=[self.audio_channel])[0], raw.info["sfreq"])

    def plays(self) -> list[Clock]:
        """Each play's clock in the saved MEG: found from the start triggers, or from the audio alone if the run has
        none."""
        alignment, recorded = self.clip.alignment, self.audio()
        nominals = self.triggers() or alignment.locate(recorded)
        return [alignment.clock(recorded, nominal) for nominal in nominals]

    def events(self) -> pd.DataFrame:
        """The recording's words and phonemes in pnpl's columns, on its plays' fitted clocks."""
        return self.events_at(self.plays())

    def events_at(self, plays: list[Clock]) -> pd.DataFrame:
        """The words and phonemes on `plays`' clocks: each play's sentences (the clip's words split at pauses)
        counted through the recording, a word's index its place in its sentence, a phoneme's in its word."""
        story = self.clip.story
        placed = [clock.place(story.events(0.0)) for clock in plays]
        per_play = story.words["sentence"].max() + 1
        shifted = [
            play.assign(sentenceidx=play[Event.SENTENCE] + shift * per_play) for shift, play in enumerate(placed)
        ]
        return Events.ordered(pd.concat(shifted), COLUMNS)


@dataclass(frozen=True)
class Kymata(Corpus[Neuromag]):
    """Kymata's recordings and the LibriBrain-shaped runs they become under `processed/kymata`."""

    @classmethod
    def under(cls, root: DataRoot) -> "Kymata":
        return cls(KYMATA.home(root), root.processed(KYMATA.dataset))

    def recordings(self, subjects: tuple[int, ...] = ()) -> Iterator[Recording]:
        """Every recording of `subjects` (all if none; numbers as `Recording.parsed` has them) with its events."""
        for fif in sorted(self.source.glob("sub-E*/meg/sub-E*_meg.fif")):
            recording = Recording(fif)
            chosen = not subjects or recording.parsed["subject"] in subjects
            if chosen and fif.stat().st_size and recording.sidecar("events.tsv").exists():
                yield recording

    def refresh(self, recording: Recording) -> list[Clock]:
        """Rewrite the events of `recording` if its run was converted, leaving its MEG alone; its plays' clocks."""
        if not self.meg(recording).exists():
            return []
        plays = recording.plays()
        recording.events_at(plays).to_csv(self.events(recording), sep="	", index=False)
        LOG.info("events -> %s", recording.name)
        return plays

    def refreshed(self, recordings: list[Recording], workers: int) -> list[Clock]:
        """Every converted recording's events rewritten in a pool; one failing is logged and does not stop the rest."""
        clocks: list[Clock] = []
        with ProcessPoolExecutor(workers) as pool:
            pending = {recording: pool.submit(self.refresh, recording) for recording in recordings}
        for recording, future in pending.items():
            error = future.exception()
            if error is None:
                clocks.extend(future.result())
            else:
                LOG.error("%s events failed: %r", recording.name, error)
        return clocks


def main() -> None:
    """Convert every recording not yet converted, or with `--events-only` rewrite the converted ones' events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-only", action="store_true", help="rewrite events of converted runs, not their MEG")
    parser.add_argument("--workers", type=int, default=None, help=f"default {WORKERS}, {EVENT_WORKERS} events-only")
    parser.add_argument("--subjects", type=int, nargs="*", default=[], help="run subject numbers (301+); default all")
    parser.add_argument("--target", type=Path, default=None, help="write the runs here instead of processed/kymata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    kymata, layout = Kymata.under(root), Neuromag.of(NEUROMAG.home(root))
    if args.target is not None:
        kymata = Kymata(kymata.source, args.target)
    recordings = list(kymata.recordings(tuple(args.subjects)))
    if args.events_only:
        clocks = kymata.refreshed(recordings, args.workers or EVENT_WORKERS)
        LOG.info("%d plays in %d runs: %s", len(clocks), len(recordings), Clock.summary(clocks))
        return
    with ProcessPoolExecutor(args.workers or WORKERS) as pool:
        written = list(pool.map(kymata.convert, recordings, [layout] * len(recordings)))
    LOG.info("%d Kymata runs under %s", len(written), kymata.target)


if __name__ == "__main__":
    main()

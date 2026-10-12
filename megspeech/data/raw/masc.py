"""MEG-MASC rewritten as LibriBrain runs: each KIT-208 recording's field mapped onto the Neuromag-306 array subject 0
was recorded on, at 250 Hz, its word and phoneme events in pnpl's columns, so every reader of a LibriBrain run reads it.

The mapping is MNE's minimum-norm field interpolation (`_map_meg_or_eeg_channels`, as `mne.Evoked.interpolate_to`
uses): both arrays sample one field outside the head, so the field each KIT gradiometer measured is re-sampled at the
Neuromag coils, through the recording's own device-to-head transform onto the head frame of MNE's Neuromag test
recording. LibriBrain's files are Maxwell-filtered with head-position correction, so one standard head position fits.

    uv run python -m megspeech.data.raw.masc   # every downloaded recording, into processed/meg_masc
"""

import argparse
import ast
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
from jaxtyping import Bool, Float
from mne.forward._field_interpolation import _map_meg_or_eeg_channels

from core.paths import DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.raw.events import Event, Kind
from megspeech.data.raw.fetch import MASC, NEUROMAG
from megspeech.data.raw.mains import Mains
from megspeech.data.raw.serialised import Corpus, RunName

LOG = logging.getLogger(__name__)

SUBJECT_OFFSET = 100
"""Added to a MASC subject's number so no MASC run shares a subject with LibriBrain's 0-32."""
MAINS = 50.0
"""The mains frequency of NYU Abu Dhabi, where MASC was recorded: its BIDS sidecars record `n/a`, and the median
spectrum of subject 1's first recording peaks at 50 Hz (5.7x its neighbours) and not at 60 (1.05x)."""
HIGHPASS = 0.1
"""LibriBrain's high-pass, in Hz."""
ORIGIN = (0.0, 0.0, 0.04)
"""The head-frame sphere origin of the field expansion, in metres: MNE's default for an adult head."""
MILLIMETRE = 1e-3
WORKERS = 4
"""Recordings converted at once: the longest holds ~3.6 GB in float64 while it is filtered."""
SOURCE = re.compile(r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-(?P<task>\d+)_meg\.con")
KEPT = tuple(Kind)
SENTENCE = "sentence"
"""The condition of a story's own sentences; `word_list` rows are the story's words shuffled, so no sentence."""
COLUMNS = [Event.KIND, Event.SEGMENT, Event.SENTENCE, Event.WORD, Event.PHONEME, Event.ONSET, Event.DURATION]


@dataclass(frozen=True)
class Neuromag:
    """The Neuromag-306 channels in LibriBrain's order, their names ascending, with MNE's test recording's geometry."""

    info: mne.Info

    @classmethod
    def of(cls, path: Path) -> "Neuromag":
        info = mne.io.read_info(path, verbose="error")
        meg = mne.pick_types(info, meg=True)
        ordered = sorted(meg, key=lambda pick: info["ch_names"][pick])
        return cls(mne.pick_info(info, ordered, verbose="error"))

    @property
    def names(self) -> list[str]:
        """The channel names as LibriBrain spells them, `MEG0111`."""
        return [name.replace(" ", "") for name in self.info["ch_names"]]

    @property
    def positions(self) -> Float[np.ndarray, "c 3"]:
        """Each channel's device-frame position in metres."""
        return np.array([channel["loc"][:3] for channel in self.info["chs"]])

    def within(self, selections: tuple[str, ...]) -> Bool[np.ndarray, " c"]:
        """Whether each channel lies in any of MNE's Vectorview `selections`."""
        listed = (name for selection in selections for name in mne.read_vectorview_selection(selection))
        chosen = {name.replace(" ", "") for name in listed}
        return np.array([name in chosen for name in self.names])

    @property
    def types(self) -> list[str]:
        return self.info.get_channel_types()

    @property
    def attrs(self) -> dict[str, object]:
        """The h5 attributes a LibriBrain run carries: its layout, rate and high-pass."""
        return {
            "channel_names": ", ".join(self.names),
            "channel_types": ", ".join(self.types),
            "sample_frequency": SFREQ,
            "highpass_cutoff": HIGHPASS,
        }

    def mapping(self, source: mne.Info) -> Float[np.ndarray, "n m"]:
        """The matrix taking `source`'s MEG channels to these channels' readings of the same field."""
        return _map_meg_or_eeg_channels(source, self.info, mode="accurate", origin=ORIGIN)


@dataclass(frozen=True)
class Recording:
    """One MASC BIDS recording: a KIT `.con` beside its markers, under its session's digitised head shape."""

    con: Path

    @cached_property
    def parsed(self) -> dict[str, int]:
        return Corpus.fields(SOURCE, self.con.name)

    @property
    def name(self) -> str:
        """The run name `Run` parses: subject past `SUBJECT_OFFSET`, the story as task `MASC<n>`, one run."""
        parsed = self.parsed
        return str(RunName(parsed["subject"] + SUBJECT_OFFSET, parsed["session"], f"MASC{parsed['task']}", 1))

    def sidecar(self, suffix: str, *, session: bool = False) -> Path:
        """The file beside the recording ending `suffix`, named after its session alone if `session`."""
        name = self.con.name
        stem = name.removesuffix("_meg.con")
        if session:
            stem = stem.rsplit("_task-", 1)[0]
        return self.con.with_name(f"{stem}_{suffix}")

    @staticmethod
    def points(path: Path) -> Float[np.ndarray, "n 3"]:
        """A FastSCAN `.pos` file's points, in metres."""
        return np.loadtxt(path, comments="%", ndmin=2) * MILLIMETRE

    def raw(self) -> mne.io.BaseRaw:
        """The recording's MEG channels with LibriBrain's band: mains notched, high-passed, at 250 Hz."""
        raw = mne.io.read_raw_kit(
            self.con,
            mrk=self.sidecar("markers.mrk"),
            elp=self.points(self.sidecar("acq-ELP_headshape.pos", session=True)),
            hsp=self.points(self.sidecar("acq-HSP_headshape.pos", session=True)),
            preload=True,
            verbose="error",
        )
        raw.pick("meg", exclude="bads")
        Mains.of(self.sidecar("meg.json"), MAINS).notch(raw)
        raw.filter(HIGHPASS, None, verbose="error")
        return raw.resample(SFREQ, verbose="error")

    def meg(self, target: Neuromag) -> Float[np.ndarray, "c t"]:
        """The recording's field at `target`'s channels, in tesla (per metre for gradiometers), float32."""
        raw = self.raw()
        return (target.mapping(raw.info) @ raw.get_data()).astype(np.float32)

    def written(self, target: Neuromag) -> tuple[Float[np.ndarray, "c t"], dict[str, object]]:
        return self.meg(target), target.attrs

    def trials(self) -> pd.DataFrame:
        """The recording's word and phoneme trials, each its BIDS `trial_type` fields beside its `onset` and
        `duration`."""
        table = pd.read_csv(self.sidecar("events.tsv"), sep="\t")
        trials = pd.DataFrame([ast.literal_eval(trial) for trial in table["trial_type"]], index=table.index)
        trials = trials.join(table[["onset", "duration"]])
        return trials[trials["kind"].isin(KEPT)]

    def heard(self) -> pd.DataFrame:
        """The heard words, in the order `events` writes them: each its `word`, the `sound` it was played in (a wav
        under the dataset's root) and its `start` second there."""
        trials = self.trials()
        heard = trials[(trials["kind"] == Kind.WORD) & trials["word"].notna()]
        return heard[["word", "sound", "start"]].reset_index(drop=True)

    @property
    def home(self) -> Path:
        """The dataset's root the recording's `sound` paths are relative to."""
        return self.con.parents[3]

    def events(self) -> pd.DataFrame:
        """The recording's words and phonemes in pnpl's columns: a story sentence's words share its `sentenceidx`, a
        word list's have none, and a phoneme counts its place in its word."""
        trials = self.trials()
        phoneme = trials["kind"] == Kind.PHONEME
        place = trials[phoneme].groupby(["sequence_id", "word_index"]).cumcount()
        return pd.DataFrame(
            {
                Event.KIND: trials["kind"],
                Event.SEGMENT: trials["word"].where(~phoneme, trials["phoneme"]),
                Event.SENTENCE: trials["sequence_id"].where(trials["condition"] == SENTENCE),
                Event.WORD: trials["word_index"],
                Event.PHONEME: place.reindex(trials.index),
                Event.ONSET: trials["onset"],
                Event.DURATION: trials["duration"],
            },
            columns=COLUMNS,
        ).reset_index(drop=True)


@dataclass(frozen=True)
class Masc(Corpus[Neuromag]):
    """MASC's downloaded recordings and the LibriBrain-shaped runs they become under `processed/meg_masc`."""

    @classmethod
    def under(cls, root: DataRoot) -> "Masc":
        return cls(MASC.home(root), root.processed(MASC.dataset))

    def recordings(self, subjects: tuple[int, ...] = ()) -> Iterator[Recording]:
        """Every recording of `subjects` (all if none) whose `.con` is non-empty, beside its events. A `.con` being
        downloaded is non-empty too: name only finished subjects while a fetch runs."""
        for con in sorted(self.source.rglob("*_meg.con")):
            recording = Recording(con)
            chosen = not subjects or recording.parsed["subject"] in subjects
            if chosen and con.stat().st_size and recording.sidecar("events.tsv").exists():
                yield recording


def main() -> None:
    """Convert every downloaded recording not yet converted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--subjects", type=int, nargs="*", default=[], help="MASC subject numbers; default all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    masc, target = Masc.under(root), Neuromag.of(NEUROMAG.home(root))
    recordings = list(masc.recordings(tuple(args.subjects)))
    with ProcessPoolExecutor(args.workers) as pool:
        written = list(pool.map(masc.convert, recordings, [target] * len(recordings)))
    LOG.info("%d MASC runs under %s", len(written), masc.target)


if __name__ == "__main__":
    main()

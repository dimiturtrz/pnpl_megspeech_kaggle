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

import h5py
import mne
import numpy as np
import pandas as pd
from jaxtyping import Bool, Float
from mne.forward._field_interpolation import _map_meg_or_eeg_channels

from core.paths import DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.raw.fetch import MASC, NEUROMAG

LOG = logging.getLogger(__name__)

SUBJECT_OFFSET = 100
"""Added to a MASC subject's number so no MASC run shares a subject with LibriBrain's 0-32."""
LINE = (60.0, 120.0)
"""New York's mains frequency and its harmonic below the 125 Hz LibriBrain keeps."""
HIGHPASS = 0.1
"""LibriBrain's high-pass, in Hz."""
ORIGIN = (0.0, 0.0, 0.04)
"""The head-frame sphere origin of the field expansion, in metres: MNE's default for an adult head."""
MILLIMETRE = 1e-3
WORKERS = 4
"""Recordings converted at once: the longest holds ~3.6 GB in float64 while it is filtered."""
SOURCE = re.compile(r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-(?P<task>\d+)_meg\.con")
KEPT = ("word", "phoneme")
SENTENCE = "sentence"
"""The condition of a story's own sentences; `word_list` rows are the story's words shuffled, so no sentence."""
SERIALISED = "serialised"
EVENTS = "events"
"""`Run.events` reads a run's events from `events/` beside the MEG's own directory."""
COLUMNS = ["kind", "segment", "sentenceidx", "wordidx", "phonemeidx", "timemeg", "duration"]


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

    def mapping(self, source: mne.Info) -> Float[np.ndarray, "n m"]:
        """The matrix taking `source`'s MEG channels to these channels' readings of the same field."""
        return _map_meg_or_eeg_channels(source, self.info, mode="accurate", origin=ORIGIN)


@dataclass(frozen=True)
class Recording:
    """One MASC BIDS recording: a KIT `.con` beside its markers, under its session's digitised head shape."""

    con: Path

    @cached_property
    def parsed(self) -> dict[str, int]:
        match = SOURCE.fullmatch(self.con.name)
        if match is None:
            raise ValueError(self.con.name)
        return {key: int(value) for key, value in match.groupdict().items()}

    @property
    def name(self) -> str:
        """The run name `Run` parses: subject past `SUBJECT_OFFSET`, the story as task `MASC<n>`, one run."""
        parsed = self.parsed
        return f"sub-{parsed['subject'] + SUBJECT_OFFSET}_ses-{parsed['session']}_task-MASC{parsed['task']}_run-1"

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
        raw.notch_filter(LINE, verbose="error")
        raw.filter(HIGHPASS, None, verbose="error")
        return raw.resample(SFREQ, verbose="error")

    def meg(self, target: Neuromag) -> Float[np.ndarray, "c t"]:
        """The recording's field at `target`'s channels, in tesla (per metre for gradiometers), float32."""
        raw = self.raw()
        return (target.mapping(raw.info) @ raw.get_data()).astype(np.float32)

    def events(self) -> pd.DataFrame:
        """The recording's words and phonemes in pnpl's columns: a story sentence's words share its `sentenceidx`, a
        word list's have none, and a phoneme counts its place in its word."""
        table = pd.read_csv(self.sidecar("events.tsv"), sep="\t")
        trials = pd.DataFrame([ast.literal_eval(trial) for trial in table["trial_type"]], index=table.index)
        trials = trials.join(table[["onset", "duration"]])
        trials = trials[trials["kind"].isin(KEPT)]
        phoneme = trials["kind"] == "phoneme"
        place = trials[phoneme].groupby(["sequence_id", "word_index"]).cumcount()
        return pd.DataFrame(
            {
                "kind": trials["kind"],
                "segment": trials["word"].where(~phoneme, trials["phoneme"]),
                "sentenceidx": trials["sequence_id"].where(trials["condition"] == SENTENCE),
                "wordidx": trials["word_index"],
                "phonemeidx": place.reindex(trials.index),
                "timemeg": trials["onset"],
                "duration": trials["duration"],
            },
            columns=COLUMNS,
        ).reset_index(drop=True)


@dataclass(frozen=True)
class Masc:
    """MASC's downloaded recordings and the LibriBrain-shaped runs they become under `processed/meg_masc`."""

    source: Path
    target: Path

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

    def meg(self, recording: Recording) -> Path:
        return self.target / SERIALISED / f"{recording.name}_meg.h5"

    def events(self, recording: Recording) -> Path:
        return self.target / EVENTS / f"{recording.name}_events.tsv"

    def convert(self, recording: Recording, target: Neuromag) -> Path:
        """Write `recording` as a LibriBrain run unless already written; its events land last, so `Run.discover`
        never finds a run whose MEG is partial."""
        meg, events = self.meg(recording), self.events(recording)
        if events.exists():
            return meg
        meg.parent.mkdir(parents=True, exist_ok=True)
        events.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(meg, "w") as stored:
            stored.create_dataset("data", data=recording.meg(target))
            stored.attrs.update(
                {
                    "channel_names": ", ".join(target.names),
                    "channel_types": ", ".join(target.types),
                    "sample_frequency": SFREQ,
                    "highpass_cutoff": HIGHPASS,
                }
            )
        recording.events().to_csv(events, sep="\t", index=False)
        LOG.info("%s -> %s", recording.con.name, recording.name)
        return meg


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

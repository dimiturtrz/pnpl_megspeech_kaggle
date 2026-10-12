"""Appleseed (OpenNeuro ds007870) rewritten as LibriBrain runs: each KIT 157-channel recording's field mapped onto the
Neuromag-306 array subject 0 was recorded on, at 250 Hz, its word and phoneme events in pnpl's columns.

The recordings carry no head-position coils (`ContinuousHeadLocalization` false), so the one static device-to-head
transform of the file stands in for head-movement correction. The cleaning is LibriBrain's in LibriBrain's order where
the data allow: bad channels, spatial cleaning (the three reference magnetometers regressed out, KIT's own noise
cancellation), mains notch, 0.1 Hz high-pass, 250 Hz. The filters are linear and per channel, so detecting the bad
channels after them rather than before them flags the same channels in the band that is kept.

    uv run python -m megspeech.data.raw.appleseed   # every downloaded recording, into processed/appleseed
"""

import argparse
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from statistics import NormalDist

import mne
import numpy as np
import pandas as pd
from jaxtyping import Bool, Float

from core.paths import DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.raw import Event, Kind
from megspeech.data.raw.fetch import APPLESEED, NEUROMAG
from megspeech.data.raw.mains import Mains
from megspeech.data.raw.masc import COLUMNS, HIGHPASS, Neuromag
from megspeech.data.raw.serialised import Corpus
from megspeech.data.raw.textgrid import TextGrid

LOG = logging.getLogger(__name__)

SUBJECT_OFFSET = 200
"""Added to an Appleseed subject's number so no run shares a subject with LibriBrain's 0-32 or MASC's 100+."""
SESSION = 1
TASK = "Appleseed"
WORKERS = 2
"""Recordings converted at once: one holds ~4.4 GB in float64 while it is filtered."""
SOURCE = re.compile(r"sub-(?P<subject>\d+)_task-Appleseed(?:_split-01)?_meg\.fif")
FALSE_ALARMS = 0.5
"""Healthy channels a bad-channel test may expect to flag per recording."""
MAD_SCALE = 1.4826
"""A normal's standard deviation in median absolute deviations."""
SENTENCE_PAUSE = 0.5
"""Seconds of silence between two words that start a new sentence. The TextGrids carry no punctuation; the silences
between their words are bimodal, short ones (below ~0.4 s) inside a clause and long ones (above ~0.6 s) between
sentences, with a trough between them, and this is the trough's middle."""
NO_PHONE = {"", "sil", "sp"}
STRESS = re.compile(r"\d+$")
"""The stress digit a CMU phone ends in."""
KIND_ORDER = {Kind.WORD: 0, Kind.PHONEME: 1}


@dataclass(frozen=True)
class Bads:
    """The channels whose spread is an outlier among the recording's: flat, railed or noisy."""

    flagged: Bool[np.ndarray, " c"]

    @staticmethod
    def limit(channels: int) -> float:
        """Robust z past which a channel is flagged: the two-sided normal quantile at which `FALSE_ALARMS` healthy
        channels of `channels` would be flagged."""
        return NormalDist().inv_cdf(1 - FALSE_ALARMS / (2 * channels))

    @classmethod
    def find(cls, meg: Float[np.ndarray, "c t"]) -> "Bads":
        """Channels whose log standard deviation lies more than `limit` robust deviations from the median; a flat
        channel's log is that of the smallest float, so it always does."""
        spread = np.log(np.maximum(meg.std(axis=1), np.finfo(np.float64).tiny))
        away = np.abs(spread - np.median(spread))
        return cls(away > cls.limit(len(meg)) * MAD_SCALE * np.median(away))


@dataclass(frozen=True)
class Cleaned:
    """The recording's good MEG channels, reference noise removed, at 250 Hz, and the record of the cleaning."""

    meg: Float[np.ndarray, "c t"]
    info: mne.Info
    bads: list[str]
    removed: float
    """The share of the channels' variance the reference regression took out."""


@dataclass(frozen=True)
class Reference:
    """The reference magnetometers' recording of the environment, which every MEG channel also picks up."""

    channels: Float[np.ndarray, "r t"]

    def clean(self, meg: Float[np.ndarray, "c t"]) -> tuple[Float[np.ndarray, "c t"], float]:
        """`meg` less its least-squares fit on the reference channels, and the share of its variance that took out."""
        coefficients = np.linalg.lstsq(self.channels.T, meg.T, rcond=None)[0]
        cleaned = meg - coefficients.T @ self.channels
        return cleaned, 1 - float(cleaned.var(axis=1).sum() / meg.var(axis=1).sum())


@dataclass(frozen=True)
class Story:
    """One story's forced alignment: words and CMU phones in seconds from the story's start."""

    grid: TextGrid

    @cached_property
    def words(self) -> pd.DataFrame:
        """The spoken words, each flagged `new` if it starts a sentence: the first, or after `SENTENCE_PAUSE` silent."""
        tier = self.grid.tiers["words"]
        words = tier[tier["text"] != ""].reset_index(drop=True)
        words["new"] = (words["start"] - words["end"].shift(fill_value=-np.inf)) >= SENTENCE_PAUSE
        words["sentence"] = words["new"].cumsum() - 1
        words[Event.WORD] = words.groupby("sentence").cumcount()
        return words

    @cached_property
    def phones(self) -> pd.DataFrame:
        """Each phone inside a word as pnpl spells it: stress dropped, lower-case, tagged B/I/E/S by its place."""
        tier = self.grid.tiers["phones"]
        heard = tier[~tier["text"].str.lower().isin(NO_PHONE)].reset_index(drop=True)
        words = self.words[["start", "end", "sentence", Event.WORD]].add_prefix("word_")
        inside = pd.merge_asof(heard, words, left_on="start", right_on="word_start")
        inside = inside[inside["start"] < inside["word_end"]]
        place = inside.groupby("word_start").cumcount()
        size = inside.groupby("word_start")["start"].transform("size")
        tag = np.select([size == 1, place == 0, place == size - 1], ["S", "B", "E"], "I")
        label = inside["text"].str.replace(STRESS, "", regex=True).str.lower() + "_" + tag
        return inside.assign(text=label, place=place)

    def events(self, onset: float) -> pd.DataFrame:
        """Word and phone rows in pnpl's columns, `timemeg` shifted by the story's `onset`."""
        words, phones = self.words, self.phones
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        Event.KIND: Kind.WORD,
                        Event.SEGMENT: words["text"],
                        Event.SENTENCE: words["sentence"],
                        Event.WORD: words[Event.WORD],
                        Event.PHONEME: np.nan,
                        Event.ONSET: words["start"] + onset,
                        Event.DURATION: words["end"] - words["start"],
                    }
                ),
                pd.DataFrame(
                    {
                        Event.KIND: Kind.PHONEME,
                        Event.SEGMENT: phones["text"],
                        Event.SENTENCE: phones["word_sentence"],
                        Event.WORD: phones["word_wordidx"],
                        Event.PHONEME: phones["place"],
                        Event.ONSET: phones["start"] + onset,
                        Event.DURATION: phones["end"] - phones["start"],
                    }
                ),
            ],
            ignore_index=True,
        )


@dataclass(frozen=True)
class Recording:
    """One Appleseed BIDS recording: the first split of a Neuromag-format `.fif`, beside its events and sidecar."""

    fif: Path

    @cached_property
    def parsed(self) -> dict[str, int]:
        match = SOURCE.fullmatch(self.fif.name)
        if match is None:
            raise ValueError(self.fif.name)
        return {key: int(value) for key, value in match.groupdict().items()}

    @property
    def name(self) -> str:
        """The run name `Run` parses: subject past `SUBJECT_OFFSET`, one session and run of task `Appleseed`."""
        return f"sub-{self.parsed['subject'] + SUBJECT_OFFSET}_ses-{SESSION}_task-{TASK}_run-1"

    def sidecar(self, suffix: str) -> Path:
        """The file beside the recording ending `suffix`, named after the task alone (no split)."""
        fif = self.fif
        stem = fif.stem.removesuffix("_meg").removesuffix("_split-01")
        return fif.with_name(f"{stem}_{suffix}")

    @property
    def stimuli(self) -> Path:
        """The dataset's `stimuli/` directory, three levels above the file."""
        return self.fif.parents[2] / "stimuli"

    def cleaned(self) -> Cleaned:
        """The recording's good MEG channels: mains notched, high-passed, 250 Hz, reference noise regressed out."""
        raw = mne.io.read_raw_fif(self.fif, preload=False, verbose="error")
        raw.pick(["meg", "ref_meg"]).load_data()
        Mains.of(self.sidecar("meg.json")).notch(raw)
        raw.filter(HIGHPASS, None, verbose="error")
        raw.resample(SFREQ, verbose="error")
        reference = Reference(raw.get_data(picks="ref_meg"))
        names = [raw.ch_names[pick] for pick in mne.pick_types(raw.info, meg=True, exclude=[])]
        flagged = Bads.find(raw.get_data(picks="meg")).flagged
        bads = sorted({*raw.info["bads"], *(name for name, bad in zip(names, flagged, strict=True) if bad)})
        raw.info["bads"] = bads
        raw.pick("meg", exclude="bads")
        meg, removed = reference.clean(raw.get_data())
        return Cleaned(meg, raw.info, bads, removed)

    def written(self, target: Neuromag) -> tuple[Float[np.ndarray, "c t"], dict[str, object]]:
        """The recording's field at `target`'s channels in tesla, float32, with its cleaning's record as attributes."""
        cleaned = self.cleaned()
        mains = Mains.of(self.sidecar("meg.json")).frequency
        LOG.info(
            "%s: bads %s, reference removed %.1f%% of variance", self.fif.name, cleaned.bads, 100 * cleaned.removed
        )
        attrs = {
            **target.attrs,
            "bad_channels": ", ".join(cleaned.bads),
            "reference_variance_removed": cleaned.removed,
            "mains": mains,
        }
        return (target.mapping(cleaned.info) @ cleaned.meg).astype(np.float32), attrs

    def events(self) -> pd.DataFrame:
        """The recording's words and phonemes in pnpl's columns: sentences are each story's words split at pauses,
        counted through the recording; a word's index counts its place in its sentence, a phoneme's in its word."""
        table = pd.read_csv(self.sidecar("events.tsv"), sep="\t", dtype={"stimulus": str})
        trials = table[(table["trial_type"] == "onset") & table["stimulus"].notna()]
        stories = [
            Story(TextGrid.read(self.stimuli / f"{stimulus}.TextGrid")).events(onset)
            for stimulus, onset in zip(trials["stimulus"], trials["onset"].astype(float), strict=True)
        ]
        before = np.cumsum([0, *(story[Event.SENTENCE].max() + 1 for story in stories[:-1])])
        events = pd.concat(
            [
                story.assign(sentenceidx=story[Event.SENTENCE] + shift)
                for story, shift in zip(stories, before, strict=True)
            ]
        )
        ordered = events.assign(order=events[Event.KIND].map(KIND_ORDER)).sort_values(
            [Event.ONSET, "order"], kind="stable"
        )
        return ordered[COLUMNS].reset_index(drop=True)


@dataclass(frozen=True)
class Appleseed(Corpus[Neuromag]):
    """Appleseed's downloaded recordings and the LibriBrain-shaped runs they become under `processed/appleseed`."""

    @classmethod
    def under(cls, root: DataRoot) -> "Appleseed":
        return cls(APPLESEED.home(root), root.processed(APPLESEED.dataset))

    def recordings(self, subjects: tuple[int, ...] = ()) -> Iterator[Recording]:
        """Every Appleseed recording of `subjects` (all if none) whose `.fif` is non-empty, beside its events. A `.fif`
        being downloaded is non-empty too: name only finished subjects while a fetch runs."""
        for fif in sorted(self.source.glob("sub-*/meg/sub-*_task-Appleseed*_meg.fif")):
            if SOURCE.fullmatch(fif.name) is None:
                continue
            recording = Recording(fif)
            chosen = not subjects or recording.parsed["subject"] in subjects
            if chosen and fif.stat().st_size and recording.sidecar("events.tsv").exists():
                yield recording


def main() -> None:
    """Convert every downloaded recording not yet converted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--subjects", type=int, nargs="*", default=[], help="Appleseed subject numbers; default all")
    parser.add_argument("--target", type=Path, default=None, help="write the runs here instead of processed/appleseed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    appleseed, layout = Appleseed.under(root), Neuromag.of(NEUROMAG.home(root))
    if args.target is not None:
        appleseed = Appleseed(appleseed.source, args.target)
    recordings = list(appleseed.recordings(tuple(args.subjects)))
    with ProcessPoolExecutor(args.workers) as pool:
        written = list(pool.map(appleseed.convert, recordings, [layout] * len(recordings)))
    LOG.info("%d Appleseed runs under %s", len(written), appleseed.target)


if __name__ == "__main__":
    main()

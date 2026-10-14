"""Every subject's recordings on disk and the partition each belongs to: the official Deep-track split for subject 0,
the pooling split for subjects 1-32."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int

from core.paths import DataRoot
from megspeech.data.constants import TAIL
from megspeech.data.raw.events import Event, Events
from megspeech.data.raw.fetch import APPLESEED, MASC
from megspeech.data.raw.serialised import NAME, RunName, Serialised
from megspeech.data.raw.sources.kymata import KYMATA
from megspeech.data.raw.sources.scans import SCANS
from megspeech.data.raw.sources.smn4lang import SMN4LANG

DEEP = 0
"""The Deep track's subject: the one the official split, the validation domains and the Kaggle test belong to."""

SOURCES = (range(DEEP, DEEP + 1), range(1, 13), range(13, 23), range(23, 33))
"""Subject-id ranges recorded under one source: subject 0 alone, then 1-12, 13-22 and 23-32 (the `GROUPS` of
`megspeech.evaluation.studies.pool.balance`, the Broad track's recording sources); any other id is its own source."""


class Column(StrEnum):
    """A column of the heard-words table each run's words are read into (`Run.words`), keyed by run."""

    RUN = "run"
    SENTENCE = "sentence"
    ONSET = "onset"
    WORD = "word"
    END = "end"
    FINISH = "finish"
    POSITION = "position"
    TARGET = "target"
    TOKEN = "token"


SHERLOCK = "Sherlock1"
"""The task subject 0's Sherlock validation and test runs and every pooled subject's runs come from."""

NO_SENTENCE = -1
"""The `Column.SENTENCE` of a word heard outside any sentence."""


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    POOL = "pool"
    """Another subject hearing the Sherlock1 ruler's text: never trained on, never a validation domain — kept for
    scoring cross-subject pooling on that text."""


HELD_OUT = {
    (SHERLOCK, 11, 2): Partition.VALIDATION,
    (SHERLOCK, 12, 2): Partition.TEST,
    ("MOCHATIMIT", 2, 1): Partition.VALIDATION,
    ("MOCHATIMIT", 3, 1): Partition.TEST,
    ("TheMoth", 29, 1): Partition.VALIDATION,
    ("TheMoth", 30, 1): Partition.TEST,
}
"""`pnpl` 0.2.0 `libribrain100/manifest.py`; every other subject-0 run trains."""

POOLED = {(SHERLOCK, 11, 1): Partition.POOL, (SHERLOCK, 12, 1): Partition.TEST}
"""Subjects 1-32 hear Sherlock1 sessions 11 and 12 once each: 11 is the text of subject 0's validation run, 12 of its
test run. Session 11 is pooled; session 12 is held out like subject 0's test run, its text never trained."""

SPLITS = {DEEP: HELD_OUT}
"""Each subject's held-out runs; a subject absent here is split by `POOLED`."""


@dataclass(frozen=True)
class Run:
    """One continuous recording: its preprocessed MEG and its word/phoneme events."""

    meg: Path

    @property
    def parsed(self) -> re.Match[str]:
        match = NAME.search(self.meg.name)
        if match is None:
            raise ValueError(self.meg.name)
        return match

    @property
    def subject(self) -> int:
        return int(self.parsed["subject"])

    @staticmethod
    def subjects(names: "pd.Series[str]") -> Int[np.ndarray, " n"]:
        """The subject of each run `name` (as `Run.name` spells it)."""
        return (names + "_").str.extract(NAME)["subject"].astype(np.int64).to_numpy()

    @staticmethod
    def tasks(names: "pd.Series[str]") -> "pd.Series[str]":
        """The task of each run `name` (as `Run.name` spells it)."""
        return (names + "_").str.extract(NAME)["task"]

    @staticmethod
    def matching(names: "pd.Series[str]", key: tuple[str, int, int]) -> Bool[np.ndarray, " n"]:
        """Whether each run `name` (as `Run.name` spells it) is of the (task, session, run) `key`."""
        parts = (names + "_").str.extract(NAME)
        task, session, run = key
        return ((parts["task"] == task) & (parts["session"] == str(session)) & (parts["run"] == str(run))).to_numpy()

    @property
    def key(self) -> tuple[str, int, int]:
        match = self.parsed
        return match["task"], int(match["session"]), int(match["run"])

    @property
    def name(self) -> str:
        task, session, run = self.key
        return str(RunName(self.subject, session, task, run))

    @staticmethod
    def signal(stored: h5py.File) -> Float[np.ndarray, "c t"]:
        """The MEG a run's open `.h5` holds, in tesla, whether stored raw or compressed."""
        return Serialised.read(stored)

    @property
    def events(self) -> Path:
        return self.meg.parent.parent / "events" / f"{self.name}_events.tsv"

    def words(self) -> pd.DataFrame:
        """The run's heard words: onset in seconds into the run, the word as spoken, its sentence in the run, and the
        `end` its window stops at — `TAIL` after its sentence's last word ends, as a Kaggle test sentence epoch does —
        and the `finish` of its own."""
        heard = Events(self.events).words(Event.SENTENCE, Event.ONSET, Event.DURATION)
        sentence = heard[Event.SENTENCE].fillna(NO_SENTENCE).to_numpy(dtype=np.int64)
        finish = heard[Event.ONSET] + heard[Event.DURATION].fillna(0.0)
        return pd.DataFrame(
            {
                Column.RUN: self.name,
                Column.ONSET: heard[Event.ONSET].to_numpy(dtype=np.float64),
                Column.WORD: heard[Event.SEGMENT].to_numpy(),
                Column.SENTENCE: sentence,
                Column.END: finish.groupby(sentence).transform("max").to_numpy(dtype=np.float64) + TAIL,
                Column.FINISH: finish.to_numpy(dtype=np.float64),
            }
        )

    @property
    def partition(self) -> Partition:
        return SPLITS.get(self.subject, POOLED).get(self.key, Partition.TRAIN)

    @classmethod
    def discover(cls, raw: Path) -> list["Run"]:
        """Every run of every subject under `raw` whose MEG and events have both finished downloading."""
        runs = (cls(meg) for meg in sorted(raw.rglob("sub-*_meg.h5")))
        return [run for run in runs if run.events.exists() and run.events.stat().st_size]

    @classmethod
    def everywhere(cls, root: DataRoot) -> list["Run"]:
        """LibriBrain's runs, then every other study's recordings rewritten as runs (`megspeech.data.raw.sources.masc`,
        `megspeech.data.raw.sources.appleseed`, `megspeech.data.raw.sources.kymata`, `megspeech.data.raw.sources.scans`,
        `megspeech.data.raw.sources.smn4lang`)."""
        studies = (MASC, APPLESEED, KYMATA, SCANS, SMN4LANG)
        rewritten = (root.processed(study.dataset) for study in studies)
        return [*cls.discover(root.raw()), *(run for processed in rewritten for run in cls.discover(processed))]

    @staticmethod
    def pooled(runs: Iterable["Run"]) -> list["Run"]:
        """The pooled runs (`Partition.POOL`) of `runs`, in subject order."""
        return sorted((run for run in runs if run.partition is Partition.POOL), key=lambda run: run.subject)

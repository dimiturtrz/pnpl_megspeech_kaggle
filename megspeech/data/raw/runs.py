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
from megspeech.data.raw.fetch import APPLESEED, MASC
from megspeech.data.raw.kymata import KYMATA
from megspeech.data.raw.scans import SCANS
from megspeech.data.raw.serialised import Serialised
from megspeech.data.raw.smn4lang import SMN4LANG

DEEP = 0
"""The Deep track's subject: the one the official split, the validation domains and the Kaggle test belong to."""

SOURCES = (range(DEEP, DEEP + 1), range(1, 13), range(13, 23), range(23, 33))
"""Subject-id ranges recorded under one source: subject 0 alone, then 1-12, 13-22 and 23-32 (the `GROUPS` of
`megspeech.evaluation.studies.pooledbalance`, the Broad track's recording sources); any other id is its own source."""


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    POOL = "pool"
    """Another subject hearing the Sherlock1 ruler's text: never trained on, never a validation domain — kept for
    scoring cross-subject pooling on that text."""


HELD_OUT = {
    ("Sherlock1", 11, 2): Partition.VALIDATION,
    ("Sherlock1", 12, 2): Partition.TEST,
    ("MOCHATIMIT", 2, 1): Partition.VALIDATION,
    ("MOCHATIMIT", 3, 1): Partition.TEST,
    ("TheMoth", 29, 1): Partition.VALIDATION,
    ("TheMoth", 30, 1): Partition.TEST,
}
"""`pnpl` 0.2.0 `libribrain100/manifest.py`; every other subject-0 run trains."""

POOLED = {("Sherlock1", 11, 1): Partition.POOL, ("Sherlock1", 12, 1): Partition.TEST}
"""Subjects 1-32 hear Sherlock1 sessions 11 and 12 once each: 11 is the text of subject 0's validation run, 12 of its
test run. Session 11 is pooled; session 12 is held out like subject 0's test run, its text never trained."""

SPLITS = {DEEP: HELD_OUT}
"""Each subject's held-out runs; a subject absent here is split by `POOLED`."""

NAME = re.compile(r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_")


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
        return f"sub-{self.subject}_ses-{session}_task-{task}_run-{run}"

    @staticmethod
    def signal(stored: h5py.File) -> Float[np.ndarray, "c t"]:
        """The MEG a run's open `.h5` holds, in tesla, whether stored raw or compressed."""
        return Serialised.read(stored)

    @property
    def events(self) -> Path:
        return self.meg.parent.parent / "events" / f"{self.name}_events.tsv"

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
        """LibriBrain's runs, then every other study's recordings rewritten as runs (`megspeech.data.raw.masc`,
        `megspeech.data.raw.appleseed`, `megspeech.data.raw.kymata`, `megspeech.data.raw.scans`,
        `megspeech.data.raw.smn4lang`)."""
        studies = (MASC, APPLESEED, KYMATA, SCANS, SMN4LANG)
        rewritten = (root.processed(study.dataset) for study in studies)
        return [*cls.discover(root.raw()), *(run for processed in rewritten for run in cls.discover(processed))]

    @staticmethod
    def pooled(runs: Iterable["Run"]) -> list["Run"]:
        """The pooled runs (`Partition.POOL`) of `runs`, in subject order."""
        return sorted((run for run in runs if run.partition is Partition.POOL), key=lambda run: run.subject)

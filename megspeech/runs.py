"""Subject 0's recordings on disk and the official Deep-track partition each belongs to."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


HELD_OUT = {
    ("Sherlock1", 11, 2): Partition.VALIDATION,
    ("Sherlock1", 12, 2): Partition.TEST,
    ("MOCHATIMIT", 2, 1): Partition.VALIDATION,
    ("MOCHATIMIT", 3, 1): Partition.TEST,
    ("TheMoth", 29, 1): Partition.VALIDATION,
    ("TheMoth", 30, 1): Partition.TEST,
}
"""`pnpl` 0.2.0 `libribrain100/manifest.py`; every other subject-0 run trains."""

NAME = re.compile(r"sub-0_ses-(?P<session>\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_")


@dataclass(frozen=True)
class Run:
    """One continuous recording: its preprocessed MEG and its word/phoneme events."""

    meg: Path

    @property
    def key(self) -> tuple[str, int, int]:
        match = NAME.search(self.meg.name)
        if match is None:
            raise ValueError(self.meg.name)
        return match["task"], int(match["session"]), int(match["run"])

    @property
    def name(self) -> str:
        task, session, run = self.key
        return f"sub-0_ses-{session}_task-{task}_run-{run}"

    @property
    def events(self) -> Path:
        return self.meg.parent.parent / "events" / f"{self.name}_events.tsv"

    @property
    def partition(self) -> Partition:
        return HELD_OUT.get(self.key, Partition.TRAIN)

    @classmethod
    def discover(cls, raw: Path) -> list["Run"]:
        """Every subject-0 run under `raw` whose MEG and events have both finished downloading."""
        runs = (cls(meg) for meg in sorted(raw.rglob("sub-0_*_meg.h5")))
        return [run for run in runs if run.events.exists() and run.events.stat().st_size]

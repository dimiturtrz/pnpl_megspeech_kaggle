from pathlib import Path

import pandas as pd
import pytest

from megspeech.data.raw.runs import Partition, Run

NAME = "sub-0_ses-11_task-Sherlock1_run-2"
POOLED = "sub-12_ses-11_task-Sherlock1_run-1"
OTHER = "sub-12_ses-12_task-Sherlock1_run-1"


def meg(root: Path, name: str = NAME) -> Path:
    path = root / "Sherlock1" / "derivatives" / "serialised" / f"{name}_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_parsed(tmp_path: Path) -> None:
    assert Run(meg(tmp_path, POOLED)).parsed["task"] == "Sherlock1"
    with pytest.raises(ValueError, match="other"):
        _ = Run(tmp_path / "other.h5").parsed


def test_subject(tmp_path: Path) -> None:
    assert [Run(meg(tmp_path, name)).subject for name in (NAME, POOLED)] == [0, 12]


def test_key(tmp_path: Path) -> None:
    assert Run(meg(tmp_path)).key == ("Sherlock1", 11, 2)
    with pytest.raises(ValueError, match="other"):
        _ = Run(tmp_path / "other.h5").key


def test_name(tmp_path: Path) -> None:
    assert Run(meg(tmp_path)).name == NAME
    assert Run(meg(tmp_path, OTHER)).name == OTHER


def test_events(tmp_path: Path) -> None:
    assert Run(meg(tmp_path)).events == tmp_path / "Sherlock1" / "derivatives" / "events" / f"{NAME}_events.tsv"


def test_partition(tmp_path: Path) -> None:
    assert Run(meg(tmp_path)).partition is Partition.VALIDATION
    assert Run(meg(tmp_path, "sub-0_ses-1_task-Sherlock1_run-1")).partition is Partition.TRAIN
    assert Run(meg(tmp_path, POOLED)).partition is Partition.POOL
    assert Run(meg(tmp_path, OTHER)).partition is Partition.TRAIN


def test_discover(tmp_path: Path) -> None:
    ready, partial = Run(meg(tmp_path)), Run(meg(tmp_path, "sub-0_ses-1_task-Sherlock1_run-1"))
    ready.events.parent.mkdir(parents=True)
    ready.events.write_text("kind\n")
    partial.events.touch()
    other = Run(meg(tmp_path, OTHER))
    other.events.write_text("kind\n")
    assert Run.discover(tmp_path) == [ready, other]


def test_subjects() -> None:
    names = pd.Series(["sub-0_ses-1_task-Sherlock1_run-1", "sub-17_ses-12_task-Sherlock1_run-1"])
    assert Run.subjects(names).tolist() == [0, 17]

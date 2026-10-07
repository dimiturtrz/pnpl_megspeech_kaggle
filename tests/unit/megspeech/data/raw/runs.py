from pathlib import Path

import pandas as pd
import pytest

from core.paths import ENV_OVERRIDE, DataRoot
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
    assert Run(meg(tmp_path, OTHER)).partition is Partition.TEST
    assert Run(meg(tmp_path, "sub-12_ses-1_task-Sherlock1_run-1")).partition is Partition.TRAIN


def test_discover(tmp_path: Path) -> None:
    ready, partial = Run(meg(tmp_path)), Run(meg(tmp_path, "sub-0_ses-1_task-Sherlock1_run-1"))
    ready.events.parent.mkdir(parents=True)
    ready.events.write_text("kind\n")
    partial.events.touch()
    other = Run(meg(tmp_path, OTHER))
    other.events.write_text("kind\n")
    assert Run.discover(tmp_path) == [ready, other]


def test_everywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    runs = [Run(meg(root.raw())), Run(meg(root.processed("meg_masc"), "sub-101_ses-0_task-MASC2_run-1"))]
    for run in runs:
        run.events.parent.mkdir(parents=True)
        run.events.write_text("kind\n")
    assert Run.everywhere(root) == runs


def test_pooled(tmp_path: Path) -> None:
    later, other, first = (Run(meg(tmp_path, name)) for name in (POOLED, OTHER, "sub-3_ses-11_task-Sherlock1_run-1"))
    assert Run.pooled([later, other, first]) == [first, later]


def test_subjects() -> None:
    names = pd.Series(["sub-0_ses-1_task-Sherlock1_run-1", "sub-17_ses-12_task-Sherlock1_run-1"])
    assert Run.subjects(names).tolist() == [0, 17]


def test_tasks() -> None:
    names = pd.Series(["sub-0_ses-1_task-Sherlock1_run-2", "sub-0_ses-3_task-TheMoth_run-1"])
    assert Run.tasks(names).tolist() == ["Sherlock1", "TheMoth"]


def test_matching() -> None:
    names = pd.Series(
        ["sub-3_ses-12_task-Sherlock1_run-1", "sub-3_ses-12_task-Sherlock1_run-2", "sub-3_ses-11_task-Sherlock1_run-1"]
    )
    assert Run.matching(names, ("Sherlock1", 12, 1)).tolist() == [True, False, False]
    assert Run.matching(names, ("TheMoth", 12, 1)).tolist() == [False, False, False]

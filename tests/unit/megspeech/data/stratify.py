from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.raw.runs import Run
from megspeech.data.stratify import BOOKS, LABELS, LOG_IOI, TEST, TRAIN, VALIDATION, Stratify

SPOKEN = [*PRIMARY.words[:3], "zzz"]
FAST, SLOW = 0.2, 0.6


def recorded(path: Path, task: str, session: int, interval: float) -> Run:
    """A run of two four-word sentences, its words `interval` seconds apart, the last word of each unknown."""
    run = Run(path / "meg" / f"sub-0_ses-{session}_task-{task}_run-1_meg.h5")
    run.events.parent.mkdir(parents=True, exist_ok=True)
    onsets = np.arange(8) * interval + np.repeat([0.0, 10.0], 4)
    rows = {"kind": "word", "segment": SPOKEN * 2, "sentenceidx": np.repeat([0, 1], 4), "timemeg": onsets}
    pd.DataFrame(rows).assign(duration=0.1).to_csv(run.events, sep="\t", index=False)
    return run


def paced(path: Path) -> tuple[Run, ...]:
    """Four fast sessions, then four slow ones."""
    return tuple(recorded(path, "Sherlock1", s, FAST if s <= 4 else SLOW) for s in range(1, 9))


def keyed(run: Run) -> int:
    return run.key[1]


def test_profile(tmp_path: Path) -> None:
    profile = Stratify.profile(recorded(tmp_path, "Sherlock2", 1, FAST), ("Sherlock1", "Sherlock2"))
    assert profile[:LOG_IOI].tolist() == [8, 2, 8]
    assert profile[LOG_IOI] == pytest.approx(6 * np.log(FAST))
    assert profile[LOG_IOI + 1 :][:2].tolist() == [6, 6]
    assert profile[LABELS][:4].tolist() == [2, 2, 2, 0]
    assert profile[BOOKS].tolist() == [0, 8]


def test_gaps() -> None:
    """A partition that matches training on every count is no gap; one at another word rate is on that axis alone."""
    held = np.array([[8.0, 2, 8, 6 * np.log(FAST), 6, 6, *[2.0] * 3, *[0.0] * 47, 8, 0]])
    same = Stratify.gaps(held, 3 * held, 0.25)
    assert same == pytest.approx(np.zeros((1, 6)), abs=1e-8)
    slower = held.copy()
    slower[0, LOG_IOI] = 6 * np.log(SLOW)
    assert np.flatnonzero(Stratify.gaps(slower, 3 * held, 0.25)[0] > 1e-8).tolist() == [1]


def test_totals() -> None:
    profiles = np.array([[1.0], [10.0], [100.0]])
    totals = Stratify.totals(profiles, np.array([[TRAIN, VALIDATION, TRAIN], [TEST, TEST, TEST]]))
    assert totals[..., 0].tolist() == [[101, 10, 0], [0, 0, 111]]


def test_cost(tmp_path: Path) -> None:
    """An assignment taking both partitions from one pace costs more than one taking each from both."""
    books = ("Sherlock1",)
    profiles = np.stack([Stratify.profile(run, books) for run in paced(tmp_path)])
    lopsided = np.array([VALIDATION, TEST, VALIDATION, TEST, TRAIN, TRAIN, TRAIN, TRAIN])
    mixed = np.array([VALIDATION, TEST, TRAIN, TRAIN, VALIDATION, TEST, TRAIN, TRAIN])
    costs = Stratify.cost(Stratify.totals(profiles, np.stack([lopsided, mixed])), 0.25, np.ones(6))
    assert costs[0] > costs[1]


def test_search(tmp_path: Path) -> None:
    books = ("Sherlock1",)
    profiles = np.stack([Stratify.profile(run, books) for run in paced(tmp_path)])
    start = np.array([VALIDATION, TEST, VALIDATION, TEST, TRAIN, TRAIN, TRAIN, TRAIN])
    cost, assigned = Stratify.search(profiles, start, np.ones(6))
    assert sorted(assigned.tolist()) == sorted(start.tolist())
    assert cost < Stratify.cost(Stratify.totals(profiles, start[None]), 0.25, np.ones(6))[0]
    assert {tuple(assigned[:4] == part) != tuple(assigned[4:] == part) for part in (VALIDATION, TEST)} <= {True}


def test_split(tmp_path: Path) -> None:
    """One fast and one slow session to each partition, the same every call; no count holds out none."""
    source = paced(tmp_path)
    val, test = Stratify.split(source, 2)
    assert [sorted(keyed(r) > len(source) // 2 for r in chosen) for chosen in (val, test)] == [[False, True]] * 2
    assert not {*val} & {*test}
    assert Stratify.split(source, 2) == (val, test)
    assert Stratify.split(source, 0) == ([], [])

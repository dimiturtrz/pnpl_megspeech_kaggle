from pathlib import Path

import pandas as pd
import pytest

from megspeech.data.raw.runs import Partition, Run
from megspeech.data.sources import Refit, Sources, Val
from megspeech.data.stratify import Stratify
from tests.unit.megspeech.data.stratify import mixed


def test_carving() -> None:
    carvings = [v.carving for v in (Val.CARVED, Val.OFFICIAL, Val.BALANCED)]
    assert [c.share for c in carvings] == [0.1, 0.0, 0.1]
    assert [c.held for c in carvings] == [(), (Partition.VALIDATION,), (Partition.VALIDATION,)]
    assert [c.tested for c in carvings] == [
        (Partition.TEST, Partition.VALIDATION),
        (Partition.TEST,),
        (Partition.TEST,),
    ]
    assert [c.tests_carved for c in carvings] == [False, False, True]


def test_stratified() -> None:
    assert [v.carving.stratified for v in (Val.CARVED, Val.OFFICIAL, Val.BALANCED)] == [True, False, True]


def test_retired_boolean_refit() -> None:
    assert [Refit(True), Refit(False), Refit("all")] == [Refit.VAL, Refit.NONE, Refit.ALL]


def named(*tasks: str) -> "pd.Series[str]":
    return pd.Series([f"sub-0_ses-1_task-{task}_run-1" for task in tasks])


def test_stratum() -> None:
    assert [Sources.stratum(t) for t in ("Sherlock3", "TheMoth", "MOCHATIMIT")] == ["Sherlock", "TheMoth", "MOCHATIMIT"]


def test_indices() -> None:
    assert Sources.indices(named("Sherlock2", "Alpha", "Sherlock1", "Alpha")).tolist() == [1, 0, 1, 0]


def test_reads() -> None:
    assert Sources.reads(named("TIMIT3", "Sherlock2", "MOCHATIMIT", "TheMoth")).tolist() == [True, False, True, False]


def test_of(tmp_path: Path) -> None:
    assert Sources.of(mixed(tmp_path)[1]) == ("Sherlock", ("Sherlock2", 1, 1))


def test_grouped(tmp_path: Path) -> None:
    groups = Sources.grouped(mixed(tmp_path))
    assert [[r.key[:2] for r in group] for group in groups] == [
        [("Sherlock1", 1), ("Sherlock1", 3), ("Sherlock2", 1)],
        [("TheMoth", 1), ("TheMoth", 2)],
    ]


def test_dealt(tmp_path: Path) -> None:
    """Keys dealt round-robin within each source, in key order."""
    runs = mixed(tmp_path)
    fold = Sources.dealt(runs, 2)
    assert [fold[r.name] for r in runs] == [1, 0, 1, 0, 0]


def test_thinned(tmp_path: Path) -> None:
    """A share keeps `ceil(share * n)` sessions of each source, nested across shares, in the given order."""
    runs = mixed(tmp_path)
    half, least = (Sources.thinned(runs, share) for share in (0.5, 0.1))
    assert [len(kept) for kept in (half, least)] == [3, 2]
    assert {r.name for r in least} <= {r.name for r in half}
    assert [r for r in runs if r in half] == half
    assert Sources.thinned(runs, 1.0) == runs


def test_picks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each source, in sorted order, beside what `Stratify.split` holds out at the ruler's share of it."""
    counts: list[int] = []

    def stand_in(source: tuple[Run, ...], count: int) -> tuple[list[Run], list[Run]]:
        counts.append(count)
        return [source[0]], []

    monkeypatch.setattr(Stratify, "split", staticmethod(stand_in))
    picks = Val.CARVED.carving.picks(mixed(tmp_path))
    assert [len(source) for source, _ in picks] == [3, 2]
    assert [val for _, (val, _) in picks] == [[picks[0][0][0]], [picks[1][0][0]]]
    assert counts == [0, 0]
    assert Val.OFFICIAL.carving.picks([]) == []

import dataclasses
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.config import METRICS, READING, DataConfig, Refit, Val
from megspeech.data.constants import RATE
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.grouping import Grouping
from megspeech.data.stratify import Picked, Stratify

TRAIN = "sub-0_ses-1_task-Sherlock1_run-1_meg.h5"
HELD = "sub-0_ses-11_task-Sherlock1_run-2_meg.h5"
OTHER = "sub-3_ses-12_task-Sherlock1_run-1_meg.h5"
POOLED = "sub-3_ses-11_task-Sherlock1_run-1_meg.h5"


OFFICIAL = DataConfig(val=Val.OFFICIAL)
CARVED = DataConfig()


def test_trains_on(tmp_path: Path) -> None:
    sherlock = Run(tmp_path / TRAIN)
    timit = Run(tmp_path / "sub-0_ses-1_task-TIMIT_run-1_meg.h5")
    held = Run(tmp_path / HELD)
    only = DataConfig(tasks=("Sherlock1",))
    assert [only.trains_on(r) for r in (sherlock, timit, held)] == [True, False, False]
    assert DataConfig().trains_on(timit)
    assert [DataConfig(val=Val.OFFICIAL, refit=Refit.VAL).trains_on(r) for r in (sherlock, held)] == [True, True]
    assert [DataConfig(refit=Refit.VAL).trains_on(r) for r in (sherlock, held)] == [True, False]
    other, pooled = Run(tmp_path / OTHER), Run(tmp_path / POOLED)
    assert [DataConfig().trains_on(r) for r in (other, pooled)] == [False, False]
    assert [DataConfig(subjects=(0, 3)).trains_on(r) for r in (sherlock, other, pooled)] == [True, False, False]
    assert not any(DataConfig(subjects=(0, 3), refit=Refit.ALL).trains_on(r) for r in (other, pooled))


def test_refitted() -> None:
    both = (Partition.VALIDATION, Partition.TEST)
    cases = [DataConfig(val=v, refit=r).refitted for v in Val for r in Refit]
    assert cases == [(), (), both, (), (Partition.VALIDATION,), both]


def runs(path: Path) -> list[Run]:
    keys = [("Sherlock1", s, 1) for s in (1, 2, 3)] + [("Sherlock1", 11, 2), ("Sherlock2", 1, 1), ("Sherlock2", 2, 1)]
    keys += [("TheMoth", s, 1) for s in (1, 2, 29)] + [("TIMIT", 1, 1)]
    return [Run(path / f"sub-0_ses-{s}_task-{task}_run-{r}_meg.h5") for task, s, r in keys]


def subjects(path: Path) -> list[Run]:
    """Subject 0's runs with two other subjects' Sherlock1 sessions 11 and 12."""
    return runs(path) + [Run(path / f"sub-{n}_ses-{s}_task-Sherlock1_run-1_meg.h5") for n in (3, 4) for s in (11, 12)]


def test_stratum() -> None:
    assert [DataConfig.stratum(t) for t in ("Sherlock3", "TheMoth", "MOCHATIMIT")] == [
        "Sherlock",
        "TheMoth",
        "MOCHATIMIT",
    ]


def keyed(domains: dict[str, list[Run]]) -> dict[str, list[tuple[str, int, int]]]:
    return {name: [r.key for r in chosen] for name, chosen in domains.items()}


def test_source(tmp_path: Path) -> None:
    assert DataConfig().source(Run(tmp_path / TRAIN)) == ("Sherlock", ("Sherlock1", 1, 1))


@pytest.fixture(autouse=True)
def first_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Stratify.split` reads every session's events; its own tests cover it, here the first sessions stand in."""

    def stand_in(source: tuple[Run, ...], count: int) -> Picked:
        return list(source[:count]), list(source[count : 2 * count])

    monkeypatch.setattr(Stratify, "split", staticmethod(stand_in))


def sessions(path: Path) -> list[Run]:
    """Twenty Sherlock1 sessions of subject 0 and an official validation run: a 10% share holds two out of each."""
    return [Run(path / f"sub-0_ses-{s}_task-Sherlock1_run-1_meg.h5") for s in range(1, 21)] + runs(path)[3:4]


def test_split(tmp_path: Path) -> None:
    """Carved val holds out `round(0.1 n)` of a source's sessions per half, disjointly; official, none."""
    held = CARVED.split(sessions(tmp_path))
    names = {p: {r.name for r in chosen} for p, chosen in held.items()}
    assert [len(names[p]) for p in (Partition.VALIDATION, Partition.TEST)] == [2, 2]
    assert names[Partition.VALIDATION].isdisjoint(names[Partition.TEST])
    assert CARVED.split(subjects(tmp_path) + sessions(tmp_path)) == CARVED.split(runs(tmp_path) + sessions(tmp_path))
    assert not any(OFFICIAL.split(sessions(tmp_path)).values())


def test_carved(tmp_path: Path) -> None:
    """Val's carved sessions are the old balanced validation half alone; official carves none; a fold only its own."""
    every = sessions(tmp_path)
    halves = CARVED.split(every)
    assert CARVED.carved(every) == halves[Partition.VALIDATION]
    assert len(CARVED.carved(every)) == 2
    assert OFFICIAL.carved(every) == []
    assert DataConfig(folds=2).carved(every) == DataConfig(folds=2).split(every)[Partition.VALIDATION]


def test_trained_split(tmp_path: Path) -> None:
    """Carved sessions never train; refitting on val takes them back, the official local test stays out until ALL."""
    every = sessions(tmp_path)
    carved = {r.name for r in CARVED.carved(every)}
    assert {r.name for r in CARVED.trained(every)}.isdisjoint(carved)
    old_test = {r.name for r in CARVED.split(every)[Partition.TEST]}
    assert old_test <= {r.name for r in CARVED.trained(every)}
    refit = DataConfig(refit=Refit.VAL).trained(every)
    assert carved <= {r.name for r in refit}
    assert all(r.partition is Partition.TRAIN for r in refit)
    assert {r.name for r in DataConfig(refit=Refit.ALL).trained(every)} == {r.name for r in every}


def test_trained_other_subjects_never(tmp_path: Path) -> None:
    """Refitting on everything still never trains subjects 1-32's validation or test partitions."""
    every = subjects(tmp_path)
    trained = DataConfig(subjects=(0, 3, 4), refit=Refit.ALL).trained(every)
    assert not [r for r in trained if r.subject]
    assert {r.name for r in trained} == {r.name for r in every if r.subject == 0}


def test_domain(tmp_path: Path) -> None:
    assert [DataConfig(val=v).domain(Run(tmp_path / TRAIN)) for v in (Val.OFFICIAL, Val.CARVED)] == [
        "Sherlock1",
        "Sherlock",
    ]
    assert DataConfig(val=Val.OFFICIAL, folds=2).domain(Run(tmp_path / TRAIN)) == "Sherlock"


def test_official(tmp_path: Path) -> None:
    every = subjects(tmp_path)
    assert {r.key for r in OFFICIAL.official(every, Partition.VALIDATION)} == {("Sherlock1", 11, 2), ("TheMoth", 29, 1)}
    assert all(r.subject == 0 for r in OFFICIAL.official(every, Partition.VALIDATION, Partition.TEST))


def test_scored(tmp_path: Path) -> None:
    """Carved val is the carved sessions alone; the local test pools pnpl's validation and test; other subjects' never
    count. Official val is pnpl's validation partition."""
    every = sessions(tmp_path)
    val, test = CARVED.scored(every, Partition.VALIDATION), CARVED.scored(every, Partition.TEST)
    assert list(val) == ["Sherlock"]
    assert len(val["Sherlock"]) == 2
    assert ("Sherlock1", 11, 2) not in {r.key for r in val["Sherlock"]}
    assert [r.key for r in test["Sherlock"]] == [("Sherlock1", 11, 2)]
    assert keyed(OFFICIAL.scored(runs(tmp_path), Partition.VALIDATION)) == {
        "Sherlock1": [("Sherlock1", 11, 2)],
        "TheMoth": [("TheMoth", 29, 1)],
    }
    assert OFFICIAL.scored(subjects(tmp_path), Partition.TEST) == {}


def test_tested(tmp_path: Path) -> None:
    official = Run(tmp_path / "sub-0_ses-12_task-Sherlock1_run-2_meg.h5")
    assert keyed(OFFICIAL.tested([*runs(tmp_path), official])) == {"Sherlock1": [("Sherlock1", 12, 2)]}
    pooled = CARVED.tested([*sessions(tmp_path), official])
    assert sorted(r.key for r in pooled["Sherlock"]) == [("Sherlock1", 11, 2), ("Sherlock1", 12, 2)]


def test_trained(tmp_path: Path) -> None:
    every = runs(tmp_path)
    assert len(OFFICIAL.trained(every)) == len(every) - 2
    assert len(DataConfig(val=Val.OFFICIAL, refit=Refit.VAL).trained(every)) == len(every)
    pooled = DataConfig(subjects=(0, 3), val=Val.OFFICIAL).trained(subjects(tmp_path))
    assert not [r for r in pooled if r.subject]


def test_thinned(tmp_path: Path) -> None:
    """A share keeps `ceil(share * n)` whole sessions of each source, nested across shares; 1 keeps every run."""
    every = sessions(tmp_path)
    full = OFFICIAL.trained(every)
    half, quarter = (DataConfig(train_share=s, val=Val.OFFICIAL).thinned(full) for s in (0.5, 0.25))
    assert len(half) == 10 and len(quarter) == 5
    assert {r.name for r in quarter} < {r.name for r in half} < {r.name for r in full}
    assert OFFICIAL.thinned(full) == full
    assert DataConfig(train_share=0.5, val=Val.OFFICIAL).trained(every) == half


def test_train_share_range() -> None:
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="train_share"):
            DataConfig(train_share=bad)


def test_trained_folds(tmp_path: Path) -> None:
    """Each member trains on all but its validation fold, and the folds tile subject 0's training sessions."""
    every = {r.name for r in OFFICIAL.trained(runs(tmp_path))}
    configs = [DataConfig(val=Val.OFFICIAL, folds=3, fold=f) for f in range(3)]
    held = [{r.name for r in config.split(runs(tmp_path))[Partition.VALIDATION]} for config in configs]
    assert [every - {r.name for r in c.trained(runs(tmp_path))} for c in configs] == held
    assert set().union(*held) == every
    assert sum(map(len, held)) == len(every)
    assert DataConfig(val=Val.OFFICIAL, folds=3, subjects=(0, 3, 4)).trained(subjects(tmp_path)) == configs[0].trained(
        runs(tmp_path)
    )


def test_folded(tmp_path: Path) -> None:
    """Validation deals the sessions test leaves round-robin into the folds; one fold keeps the pick unchanged."""
    source = tuple(Run(tmp_path / f"sub-0_ses-{s}_task-Sherlock1_run-1_meg.h5") for s in range(1, 9))
    picked = ([source[1], source[5]], [source[3], source[7]])
    assert DataConfig().folded(source, picked) == picked
    val, test = DataConfig(folds=3, fold=1).folded(source, picked)
    assert ([r.key[1] for r in val], [r.key[1] for r in test]) == ([2, 6], [4, 8])
    assert DataConfig(folds=3, fold=2).folded(source, ([], []))[0] == [source[2], source[5]]


def test_held(tmp_path: Path) -> None:
    assert keyed(OFFICIAL.held(runs(tmp_path))) == {
        "Sherlock1": [("Sherlock1", 11, 2)],
        "TheMoth": [("TheMoth", 29, 1)],
    }
    assert DataConfig(subjects=(0, 3, 4), val=Val.OFFICIAL).held(subjects(tmp_path)) == OFFICIAL.held(runs(tmp_path))


def test_ruled(tmp_path: Path) -> None:
    every = sessions(tmp_path)
    assert CARVED.ruled(every, Partition.VALIDATION) == CARVED.held(every)
    assert CARVED.ruled(every, Partition.TEST) == CARVED.tested(every)
    assert CARVED.ruled(every, Partition.TEST) != CARVED.ruled(every, Partition.VALIDATION)


def test_migrated() -> None:
    assert Val.migrated({"window": 2.0}) == {"window": 2.0, "val": Val.OFFICIAL}
    assert Val.migrated({"ruler": "official"}) == {"val": Val.OFFICIAL}
    assert Val.migrated({"ruler": "balanced"}) == {"val": Val.CARVED}
    assert Val.migrated({"share": 0.1, "stratify": True, "extra": 0}) == {"val": Val.CARVED}
    assert Val.migrated({"val": "carved"}) == {"val": "carved"}
    assert Val.migrated({"refit": True}) == {"val": Val.OFFICIAL, "refit": Refit.VAL}
    assert Val.migrated({"refit": False}) == {"val": Val.OFFICIAL, "refit": Refit.NONE}
    assert Val.migrated({"refit": "all"}) == {"val": Val.OFFICIAL, "refit": "all"}
    for retired in ({"extra": 1}, {"share": 0.1}):
        with pytest.raises(ValueError, match="retired"):
            Val.migrated(retired)


def test_share() -> None:
    assert [v.share for v in Val] == [0.1, 0.0]


def test_samples() -> None:
    assert DataConfig().samples == RATE
    assert DataConfig(window=2.5).samples == round(2.5 * RATE)
    assert DataConfig(decimate=2).samples == 125


def test_to_samples() -> None:
    assert DataConfig().to_samples(0.5) == round(0.5 * RATE)
    assert DataConfig(decimate=2).to_samples(0.5) == 62
    assert DataConfig().to_samples(0.0) == 0


def test_reading() -> None:
    at = torch.tensor([0])
    windows = DataConfig(window=2.0, lead=0.1, baseline=0.5).reading(Windows(torch.zeros(1, 10), at, at))
    assert (windows.length, windows.offset, windows.baseline) == (2 * RATE + 5, -5, round(0.5 * RATE))
    delayed = DataConfig(window=0.2, delay=0.2).reading(Windows(torch.zeros(1, 10), at, at))
    assert (delayed.length, delayed.offset) == (round(0.2 * RATE), round(0.2 * RATE))


def test_spanned() -> None:
    meg = torch.zeros(1, 400)
    at = torch.tensor([0, 100, 200])
    windows = Windows(meg, at, at + 100, following=torch.tensor([60, 100, 400]))
    assert DataConfig().spanned(windows) is windows
    assert Windows(meg, at, at + 100).following is None
    assert DataConfig(span_lag=0.0).spanned(Windows(meg, at, at + 100)).end.tolist() == [100, 200, 300]
    assert DataConfig(span_lag=0.0).spanned(windows).end.tolist() == [60, 101, 300]
    assert DataConfig(span_lag=0.2).spanned(windows).end.tolist() == [70, 110, 300]
    assert DataConfig(span_lag=0.0).reading(windows).end.tolist() == [60, 101, 300]


def test_rate() -> None:
    assert DataConfig().rate == RATE
    assert DataConfig(decimate=4).rate == 62.5


@pytest.mark.parametrize(
    ("data", "recoloured"),
    [
        (DataConfig(), False),
        (DataConfig(aligned=True), True),
        (DataConfig(subjects=(0, 101)), True),
        (DataConfig(subjects=(101,)), True),
    ],
)
def test_recoloured(data: DataConfig, recoloured: bool) -> None:
    assert data.recoloured is recoloured


def test_run_cache(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert DataConfig().run_cache(root).cache == root.processed() / "continuous50"
    fast = DataConfig(decimate=2).run_cache(root)
    assert (fast.cache.name, fast.rate) == ("continuous125", 125.0)
    assert DataConfig().cleaning is Cleaning.NONE
    assert "cleaning" in READING
    cleaned = DataConfig(cleaning=Cleaning.ICA).run_cache(root)
    assert (cleaned.cache.name, cleaned.cleaning) == ("continuous50-ica", Cleaning.ICA)


def cached(tmp_path: Path) -> DataRoot:
    """Subject 0's runs and another subject's, three channels each, cached."""
    root = DataRoot(tmp_path)
    cache = DataConfig().run_cache(root).ready()
    signal = np.random.default_rng(0).normal(5.0, 1.0, (3, 4000)).astype(np.float16)
    for run in (*runs(root.raw() / "meg"), Run(root.raw() / "meg" / OTHER)):
        for path in (run.events, run.meg):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("onset\n")
        np.save(cache / f"{run.name}.npy", signal if run.subject == 0 else 100 * signal)
    return root


@pytest.mark.parametrize(
    ("data", "aligned", "built"),
    [
        (DataConfig(), False, (False, False)),
        (DataConfig(), True, (True, False)),
        (DataConfig(aligned=True), False, (True, False)),
        (DataConfig(subspace=True), False, (False, True)),
        (DataConfig(subspace=True, aligned=True), False, (True, True)),
    ],
)
def test_geometry(tmp_path: Path, data: DataConfig, *, aligned: bool, built: tuple[bool, bool]) -> None:
    geometry = data.geometry(cached(tmp_path), "cpu", aligned=aligned)
    assert (geometry.alignment is not None, geometry.projection is not None) == built
    assert geometry.rows(3) == 3


def test_alignment(tmp_path: Path) -> None:
    root = cached(tmp_path)
    alignment = DataConfig().alignment(root, "cpu")
    assert alignment.root.shape == (3, 3)
    assert torch.allclose(alignment.median, torch.full((3, 1), 5.0), atol=0.1)
    assert not alignment.applies(runs(root.raw())[0].name)
    assert DataConfig(aligned=True).alignment(root, "cpu").applies(runs(root.raw())[0].name)


def record(directory: Path, config: dict[str, object]) -> Path:
    directory.mkdir()
    (directory / METRICS).write_text(json.dumps({"config": config}))
    return directory


def test_saved(tmp_path: Path) -> None:
    data = DataConfig(grouping=Grouping.GAP, window=2.0)
    assert DataConfig.saved(record(tmp_path / "nested", {"data": dataclasses.asdict(data)})) == data
    assert DataConfig.saved(record(tmp_path / "flat", {"epochs": 3})) == OFFICIAL


def test_shared(tmp_path: Path) -> None:
    one = record(tmp_path / "one", {})
    two = record(tmp_path / "two", {"data": {"window": 2.0}})
    scaled = record(tmp_path / "scaled", {"data": {"standardised": True}})
    assert DataConfig.shared([one, one], Grouping.GAP) == DataConfig(val=Val.OFFICIAL, grouping=Grouping.GAP)
    assert DataConfig.shared([scaled], Grouping.SENTENCE).standardised
    balanced = record(tmp_path / "balanced", {"data": {"ruler": "balanced", "refit": True}})
    legacy = DataConfig.shared([balanced], Grouping.SENTENCE)
    assert (legacy.val, legacy.refit) == (Val.CARVED, Refit.VAL)
    based = record(tmp_path / "based", {"data": {"baseline": 0.5}})
    assert DataConfig.shared([based], Grouping.SENTENCE).baseline == 0.5
    assert DataConfig.shared([record(tmp_path / "fast", {"data": {"decimate": 2}})], Grouping.SENTENCE).decimate == 2
    led = record(tmp_path / "led", {"data": {"lead": 0.1}})
    assert DataConfig.shared([led], Grouping.SENTENCE).lead == 0.1
    differing = r"runs trained at .*2.0"
    with pytest.raises(ValueError, match=differing):
        DataConfig.shared([one, two], Grouping.SENTENCE)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        DataConfig.shared([one, scaled], Grouping.SENTENCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subjects", [0, 3]),
        ("refit", "all"),
        ("features", "frozen"),
        ("read_weight", 0.5),
        ("repeat_power", 0.5),
        ("tasks", ["Sherlock1"]),
        ("subspace", True),
        ("delay", 0.2),
        ("regions", ["Left-temporal"]),
    ],
)
def test_shared_reading(tmp_path: Path, field: str, value: object) -> None:
    """Every `READING` field is carried from the members, and members differing in one cannot share a config."""
    assert field in READING
    one = record(tmp_path / "one", {})
    changed = record(tmp_path / "changed", {"data": {field: value}})
    carried = getattr(DataConfig.shared([changed, changed], Grouping.SENTENCE), field)
    assert carried == (tuple(value) if isinstance(value, list) else value)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        DataConfig.shared([one, changed], Grouping.SENTENCE)


def test_shared_folds(tmp_path: Path) -> None:
    """Bagged members together trained on every run; a single member keeps the fold it left out."""
    bagged = [record(tmp_path / f"f{f}", {"data": {"folds": 3, "fold": f}}) for f in range(3)]
    shared = DataConfig.shared(bagged, Grouping.SENTENCE)
    assert (shared.folds, shared.fold) == (1, 0)
    alone = DataConfig.shared(bagged[1:2], Grouping.SENTENCE)
    assert (alone.folds, alone.fold) == (3, 1)


def test_fold_range() -> None:
    assert DataConfig(folds=3, fold=2).fold == 2
    for fold in (-1, 3):
        with pytest.raises(ValueError, match="is not one of the 3 folds"):
            DataConfig(folds=3, fold=fold)
    with pytest.raises(ValueError, match="not both"):
        DataConfig(subspace=True, regions=("Left-temporal",))


def test_projected() -> None:
    assert not DataConfig().projected
    assert DataConfig(subspace=True).projected
    assert DataConfig(regions=("Left-temporal",)).projected


def helmet(root: DataRoot) -> None:
    """MNE's test recording reduced to three magnetometers, the first and last of them left-temporal."""
    path = NEUROMAG.home(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["MEG 0111", "MEG 0121", "MEG 0131"], RATE, "mag")
    mne.io.RawArray(np.zeros((3, 10)), info, verbose=False).save(path, verbose=False)


def test_projection(tmp_path: Path) -> None:
    root = cached(tmp_path)
    helmet(root)
    alignment = DataConfig().alignment(root, "cpu")
    assert DataConfig().projection(alignment, root) is None
    regions = DataConfig(regions=("Left-temporal",)).projection(alignment, root)
    assert regions is not None
    assert torch.equal(regions.basis, torch.eye(3)[[0, 2]])
    assert DataConfig(regions=("Left-temporal",)).geometry(root, "cpu").rows(3) == 2


def test_weight(tmp_path: Path) -> None:
    sherlock = Run(tmp_path / TRAIN)
    timit = Run(tmp_path / "sub-0_ses-1_task-TIMIT_run-1_meg.h5")
    moth = Run(tmp_path / "sub-0_ses-1_task-TheMoth_run-1_meg.h5")
    assert [DataConfig(read_weight=0.25).weight(r) for r in (sherlock, timit, moth)] == [1.0, 0.25, 1.0]
    assert [DataConfig(narrator_weight=0.5).weight(r) for r in (sherlock, timit, moth)] == [0.5, 1.0, 1.0]


def test_sources() -> None:
    names = pd.Series([f"sub-0_ses-1_task-{task}_run-1" for task in ("Sherlock2", "Alpha", "Sherlock1", "Alpha")])
    assert DataConfig.sources(names).tolist() == [1, 0, 1, 0]


def test_reads() -> None:
    names = pd.Series([f"sub-0_ses-1_task-{task}_run-1" for task in ("TIMIT3", "Sherlock2", "MOCHATIMIT", "TheMoth")])
    assert DataConfig.reads(names).tolist() == [True, False, True, False]

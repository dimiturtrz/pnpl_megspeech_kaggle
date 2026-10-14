import dataclasses
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import RUN_LEVELS, UNLEVELLED
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.config import METRICS, READING, DataConfig, Refit, Val
from megspeech.data.constants import RATE
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.sources import Sources
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
    assert cases == [(), (), both, (), (Partition.VALIDATION,), both, (), (Partition.VALIDATION,), both]


def test_admits(tmp_path: Path) -> None:
    """Subject 0 trains its training partition and what refits; another subject its training runs and `adapt`."""
    sherlock, held, other, pooled = (Run(tmp_path / name) for name in (TRAIN, HELD, OTHER, POOLED))
    assert [CARVED.admits(r) for r in (sherlock, held, other, pooled)] == [True, False, False, False]
    assert [DataConfig(val=Val.OFFICIAL, refit=Refit.VAL).admits(r) for r in (held, other)] == [True, False]
    assert [DataConfig(adapt=11).admits(r) for r in (other, pooled)] == [False, True]


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


def test_trained_balanced(tmp_path: Path) -> None:
    """A legacy balanced record withholds both carved halves from training as it did; a new carved record only val,
    an official one nothing."""
    every = sessions(tmp_path)
    balanced = DataConfig.saved(record(tmp_path / "legacy", {"data": {"ruler": "balanced"}}))
    halves = CARVED.split(every)
    names = {r.name for r in balanced.trained(every)}
    assert balanced.val is Val.BALANCED
    assert names.isdisjoint({r.name for chosen in halves.values() for r in chosen})
    assert {r.name for r in halves[Partition.TEST]} <= {r.name for r in CARVED.trained(every)}
    assert {r.name for r in OFFICIAL.trained(every)} == {r.name for r in every if r.partition is Partition.TRAIN}


def test_tested_carved(tmp_path: Path) -> None:
    every = sessions(tmp_path)
    balanced = DataConfig(val=Val.BALANCED)
    assert balanced.tested_carved(every) == CARVED.split(every)[Partition.TEST]
    assert CARVED.tested_carved(every) == OFFICIAL.tested_carved(every) == []


def test_scored_balanced(tmp_path: Path) -> None:
    """The retired ruler scored official + carved for both val and test, never pnpl's validation as test."""
    every = sessions(tmp_path)
    balanced = DataConfig(val=Val.BALANCED)
    halves = CARVED.split(every)
    val, test = balanced.held(every), balanced.tested(every)
    assert {r.name for r in val["Sherlock"]} == {r.name for r in halves[Partition.VALIDATION]} | {
        r.name for r in every if r.partition is Partition.VALIDATION
    }
    assert {r.name for r in test["Sherlock"]} == {r.name for r in halves[Partition.TEST]}


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


def test_train_share(tmp_path: Path) -> None:
    """A share trains `Sources.thinned` of the pool."""
    every = sessions(tmp_path)
    half = Sources.thinned(OFFICIAL.trained(every), 0.5)
    assert len(half) == 10
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


def test_carried() -> None:
    table = pd.DataFrame({"word": ["the", "zebra"], "run": "r", "sample": 0, "samples": 1000, "onset": [0.0, 0.1]})
    assert DataConfig(grouping=Grouping.GAP).carried(table)["word"].tolist() == ["the"]
    assert DataConfig(grouping=Grouping.SENTENCE).carried(table)["word"].tolist() == ["the", "zebra"]


def saved(tmp_path: Path, data: dict[str, object]) -> DataConfig:
    (tmp_path / METRICS).write_text(json.dumps({"config": {"data": data}}))
    return DataConfig.saved(tmp_path)


@pytest.mark.parametrize(
    ("data", "val", "refit"),
    [
        ({"window": 2.0}, Val.OFFICIAL, Refit.NONE),
        ({"ruler": "official"}, Val.OFFICIAL, Refit.NONE),
        ({"ruler": "balanced"}, Val.BALANCED, Refit.NONE),
        ({"share": 0.1, "stratify": True, "extra": 0}, Val.BALANCED, Refit.NONE),
        ({"stratify": True}, Val.OFFICIAL, Refit.NONE),
        ({"val": "carved"}, Val.CARVED, Refit.NONE),
        ({"ruler": "balanced", "val": "carved"}, Val.CARVED, Refit.NONE),
        ({"refit": True}, Val.OFFICIAL, Refit.VAL),
        ({"refit": False}, Val.OFFICIAL, Refit.NONE),
        ({"refit": "all"}, Val.OFFICIAL, Refit.ALL),
    ],
)
def test_saved_retired(tmp_path: Path, data: dict[str, object], val: Val, refit: Refit) -> None:
    config = saved(tmp_path, data)
    assert (config.val, config.refit) == (val, refit)


@pytest.mark.parametrize("retired", [{"extra": 1}, {"share": 0.1}, {"share": 0.1, "stratify": False}, {"share": 0.5}])
def test_saved_strided_refused(tmp_path: Path, retired: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="retired ruler"):
        saved(tmp_path, retired)


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
    windows = DataConfig(window=2.0).reading(Windows(torch.zeros(1, 10), at, at, baseline=3))
    assert (windows.length, windows.offset, windows.baseline) == (2 * RATE, 0, 3)
    delayed = DataConfig(window=0.2, delay=0.2).reading(Windows(torch.zeros(1, 10), at, at))
    assert (delayed.length, delayed.offset) == (round(0.2 * RATE), round(0.2 * RATE))
    assert DataConfig().reading(Windows(torch.zeros(1, 10), at, at)).segment == 0
    assert DataConfig(segment=9.0).reading(Windows(torch.zeros(1, 10), at, at)).segment == 9 * RATE


def test_segment_validated() -> None:
    with pytest.raises(ValueError, match="segment"):
        DataConfig(segment=-1.0)
    with pytest.raises(ValueError, match="segment"):
        DataConfig(segment=9.0, features="x")


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
        (DataConfig(levelled=True), False, (False, False)),
        (DataConfig(levelled=True, aligned=True), False, (True, False)),
    ],
)
def test_geometry(tmp_path: Path, data: DataConfig, *, aligned: bool, built: tuple[bool, bool]) -> None:
    root = cached(tmp_path)
    geometry = data.geometry(root, "cpu", aligned=aligned)
    assert (geometry.alignment is not None, geometry.projection is not None) == built
    assert geometry.levels is (RUN_LEVELS if data.levelled else UNLEVELLED)
    assert geometry.rows(3) == 3
    assert data.geometry(root, "cpu", aligned=aligned) is geometry


def test_built(tmp_path: Path) -> None:
    """Each call builds a fresh reference: what `geometry` keeps is the first."""
    root, data = cached(tmp_path), DataConfig(aligned=True)
    built = data.built(root, "cpu", aligned=False)
    kept = data.geometry(root, "cpu")
    assert built is not kept
    assert built.alignment is not None
    assert kept.alignment is not None
    assert torch.equal(built.alignment.root, kept.alignment.root)


def test_levelled_saved(tmp_path: Path) -> None:
    """A run saved before `levelled` existed reads its MEG unlevelled, as it trained; one saved with it, levelled."""
    old = DataConfig.saved(record(tmp_path / "old", {"data": {"aligned": False}}))
    assert not old.levelled
    assert old.geometry(cached(tmp_path / "root"), "cpu").levels is UNLEVELLED
    new = DataConfig.saved(record(tmp_path / "new", {"data": {"levelled": True}}))
    assert new.geometry(cached(tmp_path / "root"), "cpu").levels is RUN_LEVELS


def test_levelled_validated() -> None:
    with pytest.raises(ValueError, match="run levels"):
        DataConfig(levelled=True, features="x")


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


def test_partitioned(tmp_path: Path) -> None:
    """Members group by `READING` in first-seen order; bagged folds of one `folds` stay together."""
    one = record(tmp_path / "one", {})
    two = record(tmp_path / "two", {"data": {"subjects": [0, 1], "adapt": 12}})
    bagged = [record(tmp_path / f"f{f}", {"data": {"folds": 2, "fold": f}}) for f in range(2)]
    assert DataConfig.partitioned([bagged[0], one, two, bagged[1]]) == [bagged, [one], [two]]
    assert DataConfig.partitioned([two]) == [[two]]


def test_shared(tmp_path: Path) -> None:
    one = record(tmp_path / "one", {})
    two = record(tmp_path / "two", {"data": {"window": 2.0}})
    assert DataConfig.shared([one, one], Grouping.GAP) == DataConfig(val=Val.OFFICIAL, grouping=Grouping.GAP)
    balanced = record(tmp_path / "balanced", {"data": {"ruler": "balanced", "refit": True}})
    legacy = DataConfig.shared([balanced], Grouping.SENTENCE)
    assert (legacy.val, legacy.refit) == (Val.BALANCED, Refit.VAL)
    fast = record(tmp_path / "fast", {"data": {"decimate": 2}})
    assert DataConfig.shared([fast], Grouping.SENTENCE).decimate == 2
    differing = r"runs trained at .*2.0"
    with pytest.raises(ValueError, match=differing):
        DataConfig.shared([one, two], Grouping.SENTENCE)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        DataConfig.shared([one, fast], Grouping.SENTENCE)


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
        ("levelled", True),
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


def test_pooled(tmp_path: Path) -> None:
    """Without a dropped fold the pool is what trains, byte for byte."""
    every = sessions(tmp_path)
    assert CARVED.trained(every) == CARVED.pooled(every)
    assert DataConfig(crossfit=4).trained(every) == CARVED.trained(every)


def test_fold_of(tmp_path: Path) -> None:
    """The folds partition the pool by recording key, deterministically, and every subject of a key shares a fold."""
    every = subjects(tmp_path) + sessions(tmp_path)
    fold = DataConfig(subjects=(0, 3, 4)).fold_of(every, 3)
    assert fold == DataConfig(subjects=(0, 3, 4)).fold_of(list(reversed(every)), 3)
    assert set(fold) == {r.name for r in DataConfig(subjects=(0, 3, 4)).pooled(every)}
    assert set(fold.values()) == {0, 1, 2}
    by_key: dict[tuple[str, int, int], set[int]] = {}
    for r in every:
        if r.name in fold:
            by_key.setdefault(r.key, set()).add(fold[r.name])
    assert all(len(folds) == 1 for folds in by_key.values())


def test_trained_dropped(tmp_path: Path) -> None:
    """Dropping fold j removes exactly its sessions from training; val, the local test and the other folds stay."""
    every = sessions(tmp_path)
    pool = {r.name for r in CARVED.trained(every)}
    fold = CARVED.fold_of(every, 3)
    dropped = [DataConfig(crossfit=3, dropped=j) for j in range(3)]
    for j, config in enumerate(dropped):
        assert {r.name for r in config.trained(every)} == pool - {n for n, f in fold.items() if f == j}
        assert config.held(every) == CARVED.held(every)
        assert config.tested(every) == CARVED.tested(every)
    assert sorted(n for c in dropped for n in pool - {r.name for r in c.trained(every)}) == sorted(pool)


def test_dropped_range() -> None:
    assert DataConfig(crossfit=3, dropped=2).dropped == 2
    for crossfit, dropped in ((3, 3), (3, -2), (1, 0)):
        with pytest.raises(ValueError, match="crossfit folds"):
            DataConfig(crossfit=crossfit, dropped=dropped)
    assert "dropped" not in READING


def test_cut() -> None:
    assert DataConfig().cut() == (RATE, 0)
    assert DataConfig(window=2.0).cut() == (2 * RATE, 0)
    assert DataConfig(window=0.2, delay=0.2).cut() == (round(0.2 * RATE), round(0.2 * RATE))


def test_trains_on_adapt(tmp_path: Path) -> None:
    """`adapt` trains that Sherlock1 session of the listed subjects 1-32 alone: not the other session, not an unlisted
    subject, not by default."""
    other, pooled = Run(tmp_path / OTHER), Run(tmp_path / POOLED)
    outsider = Run(tmp_path / OTHER.replace("sub-3", "sub-4"))
    twelve, eleven = DataConfig(subjects=(0, 3), adapt=12), DataConfig(subjects=(0, 3), adapt=11)
    assert [twelve.trains_on(r) for r in (other, pooled, outsider)] == [True, False, False]
    assert [eleven.trains_on(r) for r in (other, pooled, outsider)] == [False, True, False]
    assert [DataConfig(subjects=(0,), adapt=12).trains_on(r) for r in (other, pooled)] == [False, False]
    assert [DataConfig(subjects=(0, 3)).trains_on(r) for r in (other, pooled)] == [False, False]
    refit = DataConfig(subjects=(0, 3), adapt=12, refit=Refit.ALL)
    assert [refit.trains_on(r) for r in (other, pooled, outsider)] == [True, True, False]


def test_adapted() -> None:
    """`adapt` alone trains its session; a refit of everything adds the ruler's; no `adapt` trains neither."""
    assert DataConfig(adapt=12).adapted == (12,)
    assert DataConfig(adapt=11, refit=Refit.VAL).adapted == (11,)
    assert DataConfig(adapt=12, refit=Refit.ALL).adapted == (11, 12)
    assert DataConfig(refit=Refit.ALL).adapted == ()


def test_trained_adapt_discovered(tmp_path: Path) -> None:
    """Over discovered runs, `adapt=12` selects exactly the listed subjects' session-12 runs."""
    for subject in range(1, 5):
        for session in (11, 12):
            name = f"sub-{subject}_ses-{session}_task-Sherlock1_run-1"
            (tmp_path / "events").mkdir(exist_ok=True)
            (tmp_path / "meg").mkdir(exist_ok=True)
            (tmp_path / "meg" / f"{name}_meg.h5").write_bytes(b"x")
            (tmp_path / "events" / f"{name}_events.tsv").write_text("onset")
    runs = Run.discover(tmp_path)
    trained = DataConfig(subjects=(1, 2, 3), adapt=12).trained(runs)
    assert sorted(r.name for r in trained) == [f"sub-{s}_ses-12_task-Sherlock1_run-1" for s in (1, 2, 3)]


def test_muted() -> None:
    """Only the `adapt` session of subjects 1-32 is muted, and `adapt_context` or no `adapt` mutes nothing."""
    names = pd.Series(
        [
            "sub-3_ses-12_task-Sherlock1_run-1",
            "sub-3_ses-11_task-Sherlock1_run-1",
            "sub-0_ses-12_task-Sherlock1_run-1",
            "sub-0_ses-1_task-Sherlock1_run-1",
        ]
    )
    assert DataConfig(subjects=(0, 3), adapt=12).muted(names).tolist() == [True, False, False, False]
    assert DataConfig(subjects=(0, 3), adapt=12, adapt_context=True).muted(names).tolist() == [False] * 4
    assert DataConfig().muted(names).tolist() == [False] * 4
    refit = DataConfig(subjects=(0, 3), adapt=12, refit=Refit.ALL)
    assert refit.muted(names).tolist() == [True, True, False, False]


def test_muted_context_subjects() -> None:
    """`context_subjects` mutes every run of the subjects outside it, on top of the `adapt` rule; empty mutes none."""
    names = pd.Series(
        [
            "sub-3_ses-12_task-Sherlock1_run-1",
            "sub-3_ses-1_task-Sherlock1_run-1",
            "sub-0_ses-1_task-Sherlock1_run-1",
            "sub-301_ses-1_task-Sherlock1_run-1",
        ]
    )
    assert DataConfig(subjects=(0, 3, 301), context_subjects=(0,)).muted(names).tolist() == [True, True, False, True]
    assert DataConfig(subjects=(0, 3), adapt=12, adapt_context=True, context_subjects=(3,)).muted(names).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert DataConfig(subjects=(0, 3), adapt=12, context_subjects=(0, 3)).muted(names).tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert DataConfig(context_subjects=()).muted(names).tolist() == [False] * 4


def test_reading_fields_retired() -> None:
    assert Tree(DataConfig).load({"standardised": False, "lead": 0.0, "baseline": 0.0}) == DataConfig()
    for name, value, killer in (
        ("standardised", True, "7gf.25.4"),
        ("lead", 0.1, "7gf.38"),
        ("baseline", 0.5, "7gf.157.5"),
    ):
        with pytest.raises(ValueError, match=killer):
            Tree(DataConfig).load({name: value})

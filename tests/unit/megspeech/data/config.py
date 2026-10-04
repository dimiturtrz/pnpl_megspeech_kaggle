import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.config import METRICS, READING, DataConfig
from megspeech.data.constants import RATE
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import Grouping

TRAIN = "sub-0_ses-1_task-Sherlock1_run-1_meg.h5"
HELD = "sub-0_ses-11_task-Sherlock1_run-2_meg.h5"
OTHER = "sub-3_ses-12_task-Sherlock1_run-1_meg.h5"
POOLED = "sub-3_ses-11_task-Sherlock1_run-1_meg.h5"


def test_trains_on(tmp_path: Path) -> None:
    sherlock = Run(tmp_path / TRAIN)
    timit = Run(tmp_path / "sub-0_ses-1_task-TIMIT_run-1_meg.h5")
    held = Run(tmp_path / HELD)
    only = DataConfig(tasks=("Sherlock1",))
    assert [only.trains_on(r) for r in (sherlock, timit, held)] == [True, False, False]
    assert DataConfig().trains_on(timit)
    assert [DataConfig(refit=True).trains_on(r) for r in (sherlock, held)] == [True, True]
    other, pooled = Run(tmp_path / OTHER), Run(tmp_path / POOLED)
    assert [DataConfig().trains_on(r) for r in (other, pooled)] == [False, False]
    assert [DataConfig(subjects=(0, 3)).trains_on(r) for r in (sherlock, other, pooled)] == [True, False, False]
    assert not any(DataConfig(subjects=(0, 3), refit=True).trains_on(r) for r in (other, pooled))


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


def test_extras(tmp_path: Path) -> None:
    keys = [r.key for r in DataConfig(extra=3).extras(runs(tmp_path))]
    assert keys == [("Sherlock1", 3, 1), ("Sherlock2", 2, 1), ("Sherlock1", 2, 1), ("TheMoth", 2, 1), ("TheMoth", 1, 1)]
    assert DataConfig().extras(runs(tmp_path)) == []
    assert DataConfig(extra=3, subjects=(0, 3, 4)).extras(subjects(tmp_path)) == DataConfig(extra=3).extras(
        runs(tmp_path)
    )


def keyed(domains: dict[str, list[Run]]) -> dict[str, list[tuple[str, int, int]]]:
    return {name: [r.key for r in chosen] for name, chosen in domains.items()}


def test_source(tmp_path: Path) -> None:
    assert DataConfig().source(Run(tmp_path / TRAIN)) == ("Sherlock", ("Sherlock1", 1, 1))


def test_split(tmp_path: Path) -> None:
    """Each source holds out one session per stride for each partition; a source too small for one holds out none."""
    held_out = {p.value: [r.key for r in chosen] for p, chosen in DataConfig(share=0.4).split(runs(tmp_path)).items()}
    assert held_out == {
        "validation": [("Sherlock1", 1, 1), ("Sherlock2", 1, 1), ("TheMoth", 1, 1)],
        "test": [("Sherlock1", 2, 1), ("Sherlock2", 2, 1), ("TheMoth", 2, 1)],
    }
    assert DataConfig(share=0.4).split(subjects(tmp_path)) == DataConfig(share=0.4).split(runs(tmp_path))
    assert not any(DataConfig().split(runs(tmp_path)).values())


def test_strided(tmp_path: Path) -> None:
    """One session per stride, validation a quarter in, test three quarters in."""
    source = tuple(Run(tmp_path / f"sub-0_ses-{s}_task-Sherlock1_run-1_meg.h5") for s in range(1, 9))
    val, test = DataConfig.strided(source, 2)
    assert ([r.key[1] for r in val], [r.key[1] for r in test]) == ([2, 6], [4, 8])
    assert DataConfig.strided(source, 0) == ([], [])


def test_trained_split(tmp_path: Path) -> None:
    """Held-out sessions never train; refitting takes the validation ones back, never the test ones."""
    every = runs(tmp_path)
    assert {r.key for r in DataConfig(share=0.4).trained(every)} == {("Sherlock1", 3, 1), ("TIMIT", 1, 1)}
    refit = {r.key for r in DataConfig(share=0.4, refit=True).trained(every)}
    assert refit == {r.key for r in every} - {("Sherlock1", 2, 1), ("Sherlock2", 2, 1), ("TheMoth", 2, 1)}


def test_domain(tmp_path: Path) -> None:
    assert [DataConfig(share=s).domain(Run(tmp_path / TRAIN)) for s in (0.0, 0.1)] == ["Sherlock1", "Sherlock"]
    assert DataConfig(folds=2).domain(Run(tmp_path / TRAIN)) == "Sherlock"


def test_scored(tmp_path: Path) -> None:
    """Under a share, the official runs join their source's held-out sessions; other subjects' test runs never count."""
    assert keyed(DataConfig(share=0.4).scored(runs(tmp_path), Partition.VALIDATION)) == {
        "Sherlock": [("Sherlock1", 11, 2), ("Sherlock1", 1, 1), ("Sherlock2", 1, 1)],
        "TheMoth": [("TheMoth", 29, 1), ("TheMoth", 1, 1)],
    }
    assert DataConfig().scored(subjects(tmp_path), Partition.TEST) == {}


def test_tested(tmp_path: Path) -> None:
    tested = DataConfig(share=0.4).tested(subjects(tmp_path))
    assert keyed(tested) == {"Sherlock": [("Sherlock1", 2, 1), ("Sherlock2", 2, 1)], "TheMoth": [("TheMoth", 2, 1)]}
    official = Run(tmp_path / "sub-0_ses-12_task-Sherlock1_run-2_meg.h5")
    assert keyed(DataConfig().tested([*runs(tmp_path), official])) == {"Sherlock1": [("Sherlock1", 12, 2)]}


def test_trained(tmp_path: Path) -> None:
    every = runs(tmp_path)
    assert len(DataConfig().trained(every)) == len(every) - 2
    trained = {r.key for r in DataConfig(extra=1).trained(every)}
    assert {("Sherlock1", 3, 1), ("TheMoth", 2, 1)}.isdisjoint(trained)
    assert len(trained) == len(every) - 4
    assert len(DataConfig(extra=1, refit=True).trained(every)) == len(every)
    assert DataConfig(extra=1).trained(subjects(tmp_path)) == DataConfig(extra=1).trained(every)
    pooled = DataConfig(subjects=(0, 3)).trained(subjects(tmp_path))
    assert not [r for r in pooled if r.subject]


def test_trained_folds(tmp_path: Path) -> None:
    """Each member trains on all but its validation fold, and the folds tile subject 0's training sessions."""
    every = {r.name for r in DataConfig().trained(runs(tmp_path))}
    configs = [DataConfig(folds=3, fold=f) for f in range(3)]
    held = [{r.name for r in config.split(runs(tmp_path))[Partition.VALIDATION]} for config in configs]
    assert [every - {r.name for r in c.trained(runs(tmp_path))} for c in configs] == held
    assert set().union(*held) == every
    assert sum(map(len, held)) == len(every)
    assert DataConfig(folds=3, subjects=(0, 3, 4)).trained(subjects(tmp_path)) == configs[0].trained(runs(tmp_path))


def test_folded(tmp_path: Path) -> None:
    """Validation deals the sessions test leaves round-robin into the folds; one fold keeps the pick unchanged."""
    source = tuple(Run(tmp_path / f"sub-0_ses-{s}_task-Sherlock1_run-1_meg.h5") for s in range(1, 9))
    picked = DataConfig.strided(source, 2)
    assert DataConfig().folded(source, picked) == picked
    val, test = DataConfig(folds=3, fold=1).folded(source, picked)
    assert ([r.key[1] for r in val], [r.key[1] for r in test]) == ([2, 6], [4, 8])
    assert DataConfig(folds=3, fold=2).folded(source, ([], []))[0] == [source[2], source[5]]


def test_held(tmp_path: Path) -> None:
    assert {k: [r.key for r in v] for k, v in DataConfig().held(runs(tmp_path)).items()} == {
        "Sherlock1": [("Sherlock1", 11, 2)],
        "TheMoth": [("TheMoth", 29, 1)],
    }
    held = DataConfig(extra=1).held(runs(tmp_path))
    assert list(held) == ["Sherlock1", "TheMoth", "Sherlock+", "TheMoth+"]
    assert [r.key for r in held["Sherlock+"]] == [("Sherlock1", 3, 1)]
    assert DataConfig(extra=1, subjects=(0, 3, 4)).held(subjects(tmp_path)) == held


def test_ruled(tmp_path: Path) -> None:
    config = DataConfig(share=0.4)
    every = subjects(tmp_path)
    assert config.ruled(every, Partition.VALIDATION) == config.held(every)
    assert config.ruled(every, Partition.TEST) == config.tested(every)
    assert config.ruled(every, Partition.TEST) != config.ruled(every, Partition.VALIDATION)


def test_official_split() -> None:
    folded = DataConfig(share=0.1, stratify=True, folds=5, fold=2, window=0.2).official_split
    assert (folded.share, folded.stratify, folded.folds, folded.fold, folded.window) == (0.0, False, 1, 0, 0.2)


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
    assert DataConfig.saved(record(tmp_path / "flat", {"epochs": 3})) == DataConfig()


def test_shared(tmp_path: Path) -> None:
    one = record(tmp_path / "one", {})
    two = record(tmp_path / "two", {"data": {"window": 2.0}})
    scaled = record(tmp_path / "scaled", {"data": {"standardised": True}})
    assert DataConfig.shared([one, one], Grouping.GAP) == DataConfig(grouping=Grouping.GAP)
    assert DataConfig.shared([scaled], Grouping.SENTENCE).standardised
    widened = record(tmp_path / "widened", {"data": {"extra": 2}})
    assert DataConfig.shared([widened], Grouping.SENTENCE).extra == 2
    based = record(tmp_path / "based", {"data": {"baseline": 0.5}})
    assert DataConfig.shared([based], Grouping.SENTENCE).baseline == 0.5
    assert DataConfig.shared([record(tmp_path / "fast", {"data": {"decimate": 2}})], Grouping.SENTENCE).decimate == 2
    led = record(tmp_path / "led", {"data": {"lead": 0.1}})
    assert DataConfig.shared([led], Grouping.SENTENCE).lead == 0.1
    differing = r"False, 1.0, False, 0, 0.0, False, 0.0, 0.0, 0.0, 5, .*False, 2.0, "
    with pytest.raises(ValueError, match=differing):
        DataConfig.shared([one, two], Grouping.SENTENCE)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        DataConfig.shared([one, scaled], Grouping.SENTENCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subjects", [0, 3]),
        ("refit", True),
        ("features", "frozen"),
        ("read_weight", 0.5),
        ("repeat_power", 0.5),
        ("tasks", ["Sherlock1"]),
        ("subspace", True),
        ("delay", 0.2),
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


def test_weight(tmp_path: Path) -> None:
    sherlock = Run(tmp_path / TRAIN)
    timit = Run(tmp_path / "sub-0_ses-1_task-TIMIT_run-1_meg.h5")
    moth = Run(tmp_path / "sub-0_ses-1_task-TheMoth_run-1_meg.h5")
    assert [DataConfig(read_weight=0.25).weight(r) for r in (sherlock, timit, moth)] == [1.0, 0.25, 1.0]
    assert [DataConfig(narrator_weight=0.5).weight(r) for r in (sherlock, timit, moth)] == [0.5, 1.0, 1.0]

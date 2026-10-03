import dataclasses
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.continuous import DIRECTORY, RATE, WORDS, Continuous
from megspeech.data.corpus import METRICS, READING, Corpus, DataConfig
from megspeech.data.grouping import Grouping
from megspeech.data.runs import CHANNELS, Run
from megspeech.data.vocabulary import IGNORE, PRIMARY
from megspeech.data.windows import Windows

SECONDS = 4
CPU = torch.device("cpu")
TRAIN = "sub-0_ses-1_task-Sherlock1_run-1_meg.h5"
HELD = "sub-0_ses-11_task-Sherlock1_run-2_meg.h5"
OTHER = "sub-3_ses-12_task-Sherlock1_run-1_meg.h5"
POOLED = "sub-3_ses-11_task-Sherlock1_run-1_meg.h5"
THE, A = PRIMARY.labels(["the", "a"]).tolist()


def tokens(*words: str) -> tuple[int, ...]:
    return tuple(pd.util.hash_array(np.array(words, dtype=object)).view(np.int64).tolist())


class Root:
    def __init__(self, raw: Path) -> None:
        self.path = raw

    def raw(self) -> Path:
        return self.path

    def processed(self) -> Path:
        return self.path / "processed"


def corpus_root(tmp_path: Path) -> DataRoot:
    """A train run heard `the zebra a | was` (`was` too late for a window) and a validation run heard `the a`."""
    cache = tmp_path / "processed" / DIRECTORY.format(rate=RATE)
    cache.mkdir(parents=True)
    heard = {
        TRAIN: [("the", 0.5, 0, 2.2), ("zebra", 0.9, 0, 2.2), ("a", 1.3, 0, 2.2), ("was", 3.9, 1, 4.9)],
        HELD: [("the", 0.5, 0, 2.0), ("a", 1.0, 0, 2.0)],
        OTHER: [("the", 0.5, 0, 2.0)],
    }
    rows: list[tuple[str, float, str, int, float]] = []
    for number, (name, words) in enumerate(heard.items()):
        run = Run(tmp_path / name)
        run.meg.touch()
        run.events.parent.mkdir(parents=True, exist_ok=True)
        run.events.write_text("kind")
        signal = np.full((CHANNELS, SECONDS * RATE), number, np.float16)
        signal[:, ::2] += 1
        np.save(cache / f"{run.name}.npy", signal)
        rows += [(run.name, onset, word, sentence, end) for word, onset, sentence, end in words]
    pd.DataFrame(rows, columns=["run", "onset", "word", "sentence", "end"]).to_csv(cache / WORDS, sep="\t", index=False)
    return cast("DataRoot", Root(tmp_path))


def source(tmp_path: Path) -> Continuous:
    names = [Run(tmp_path / name).name for name in (TRAIN, HELD)]
    return Continuous.load(corpus_root(tmp_path).processed() / DIRECTORY.format(rate=RATE), CPU, names)


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
    assert [DataConfig(subjects=(0, 3)).trains_on(r) for r in (sherlock, other, pooled)] == [True, True, False]
    assert not DataConfig(subjects=(0, 3), refit=True).trains_on(pooled)


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


def test_trained(tmp_path: Path) -> None:
    every = runs(tmp_path)
    assert len(DataConfig().trained(every)) == len(every) - 2
    trained = {r.key for r in DataConfig(extra=1).trained(every)}
    assert {("Sherlock1", 3, 1), ("TheMoth", 2, 1)}.isdisjoint(trained)
    assert len(trained) == len(every) - 4
    assert len(DataConfig(extra=1, refit=True).trained(every)) == len(every)
    assert DataConfig(extra=1).trained(subjects(tmp_path)) == DataConfig(extra=1).trained(every)
    pooled = DataConfig(subjects=(0, 3)).trained(subjects(tmp_path))
    assert [r.name for r in pooled if r.subject] == ["sub-3_ses-12_task-Sherlock1_run-1"]


def test_trained_folds(tmp_path: Path) -> None:
    """Folds partition the training runs: each member drops one fold, and the dropped folds tile the whole."""
    every = {r.name for r in DataConfig().trained(runs(tmp_path))}
    kept = [{r.name for r in DataConfig(folds=3, fold=f).trained(runs(tmp_path))} for f in range(3)]
    dropped = [every - k for k in kept]
    assert all(k < every for k in kept)
    assert set().union(*dropped) == every
    assert sum(map(len, dropped)) == len(every)
    assert [DataConfig(folds=3, fold=f).trained(subjects(tmp_path)) for f in range(3)] == [
        DataConfig(folds=3, fold=f).trained(runs(tmp_path)) for f in range(3)
    ]


def test_held(tmp_path: Path) -> None:
    assert {k: [r.key for r in v] for k, v in DataConfig().held(runs(tmp_path)).items()} == {
        "Sherlock1": [("Sherlock1", 11, 2)],
        "TheMoth": [("TheMoth", 29, 1)],
    }
    held = DataConfig(extra=1).held(runs(tmp_path))
    assert list(held) == ["Sherlock1", "TheMoth", "Sherlock+", "TheMoth+"]
    assert [r.key for r in held["Sherlock+"]] == [("Sherlock1", 3, 1)]
    assert DataConfig(extra=1, subjects=(0, 3, 4)).held(subjects(tmp_path)) == held


def test_samples() -> None:
    assert DataConfig().samples == RATE
    assert DataConfig(window=2.5).samples == round(2.5 * RATE)
    assert DataConfig(decimate=2).samples == 125


def test_centring() -> None:
    assert DataConfig().centring == 0
    assert DataConfig(baseline=0.5).centring == round(0.5 * RATE)
    assert DataConfig(baseline=0.5, decimate=2).centring == 62


def test_leading() -> None:
    assert DataConfig().leading == 0
    assert DataConfig(lead=0.1).leading == round(0.1 * RATE)


def test_reading() -> None:
    at = torch.tensor([0])
    windows = DataConfig(window=2.0, lead=0.1, baseline=0.5).reading(Windows(torch.zeros(1, 10), at, at))
    assert (windows.length, windows.offset, windows.baseline) == (2 * RATE + 5, -5, round(0.5 * RATE))


def test_rate() -> None:
    assert DataConfig().rate == RATE
    assert DataConfig(decimate=4).rate == 62.5


def test_decimator(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    assert DataConfig().decimator(root).cache == root.processed() / "continuous50"
    fast = DataConfig(decimate=2).decimator(root)
    assert (fast.cache.name, fast.rate) == ("continuous125", 125.0)


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
    with pytest.raises(ValueError, match=r"\(1.0, False, 0, 0.0, 0.0, 5, .*\(2.0, False, 0, 0.0, 0.0, 5, "):
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
    assert [DataConfig(read_weight=0.25).weight(r) for r in (sherlock, timit)] == [1.0, 0.25]


def test_words(tmp_path: Path) -> None:
    table = source(tmp_path).words
    sentence = Corpus.words(table, Grouping.SENTENCE)
    assert sentence["word"].tolist() == ["the", "zebra", "a", "the", "a"]
    assert sentence["target"].tolist() == [THE, IGNORE, A, THE, A]
    assert Corpus.words(table, Grouping.GAP)["word"].tolist() == ["the", "a", "the", "a"]


def test_token() -> None:
    assert Corpus.token(pd.Series(["The", "the", "a"])).tolist() == list(tokens("the", "the", "a"))


def test_split(tmp_path: Path) -> None:
    loaded = source(tmp_path)
    words = Corpus.words(loaded.words, Grouping.SENTENCE)
    split = Corpus.split(loaded, words, DataConfig(), np.arange(5.0))
    assert split.y.tolist() == [THE, IGNORE, A, THE, A]
    assert split.index.tolist() == [[0, 1, 2] + [-1] * 6, [3, 4] + [-1] * 7]
    assert split.weight.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    window = split.windows(torch.tensor([4]))
    assert window.shape == (1, CHANNELS, RATE)
    assert window.float().abs().tolist() == [[[0.5] * RATE] * CHANNELS]
    first = torch.tensor([0])
    long = Corpus.split(loaded, words, DataConfig(window=2.0), np.ones(5)).windows(first)
    assert long.shape == (1, CHANNELS, 2 * RATE)
    tail = round(2.2 * RATE) - round(0.5 * RATE)
    assert long[..., tail:].eq(0).all()
    assert long[..., :tail].ne(0).all()
    headed = Corpus.split(loaded, words, DataConfig(baseline=0.02), np.ones(5)).windows(torch.tensor([4]))
    assert headed[..., 0].float().eq(0).all()
    led = Corpus.split(loaded, words, DataConfig(lead=0.1), np.ones(5)).windows(torch.tensor([0, 1]))
    assert led.shape == (2, CHANNELS, RATE + 5)
    assert led[0, :, :5].eq(0).all()
    assert led[1, :, :5].ne(0).all()
    gap = DataConfig(grouping=Grouping.GAP, window=2.0)
    assert Corpus.split(loaded, Corpus.words(loaded.words, Grouping.GAP), gap, np.ones(4)).windows(first).ne(0).all()


def test_domains(tmp_path: Path) -> None:
    loaded = source(tmp_path)
    words = Corpus.words(loaded.words, Grouping.SENTENCE)
    validation = Corpus.domains(loaded, words, {"Sherlock1": [Run(tmp_path / HELD)], "Sherlock+": []}, DataConfig())
    assert list(validation) == ["Sherlock1", "Sherlock+"]
    assert validation["Sherlock1"].y.tolist() == [THE, A]
    assert validation["Sherlock+"].y.tolist() == []


def test_heard(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    found = Run.discover(root.raw())
    heard = Corpus.heard(DataConfig().decimator(root).cache, found, DataConfig())
    assert heard["word"].tolist() == ["the", "zebra", "a"]
    assert Corpus.heard(DataConfig().decimator(root).cache, found, DataConfig(extra=1)).empty
    pooled = Corpus.heard(DataConfig().decimator(root).cache, found, DataConfig(subjects=(0, 3)))
    assert pooled["word"].tolist() == ["the", "zebra", "a", "the"]
    assert pooled["weight"].tolist() == [1.0] * 4


@pytest.mark.parametrize(
    ("data", "words"),
    [
        (DataConfig(refit=True), ["the", "a", "the", "zebra", "a"]),
        (DataConfig(tasks=("Sherlock2",)), []),
        (DataConfig(folds=2, fold=0), []),
        (DataConfig(folds=2, fold=1), ["the", "zebra", "a"]),
    ],
)
def test_heard_trained(tmp_path: Path, data: DataConfig, words: list[str]) -> None:
    """`heard` is the text of exactly the runs training reads, whatever selects them."""
    root = corpus_root(tmp_path)
    found = Run.discover(root.raw())
    assert Corpus.heard(data.decimator(root).cache, found, data)["word"].tolist() == words
    assert {r.name for r in data.trained(found)} >= set(Corpus.heard(data.decimator(root).cache, found, data)["run"])


@pytest.mark.parametrize("data", [DataConfig(), DataConfig(repeat_power=0.5), DataConfig(subjects=(0, 3))])
def test_prior(tmp_path: Path, data: DataConfig) -> None:
    """The prior read from the word table alone is the one training's split counts."""
    root = corpus_root(tmp_path)
    heard = Corpus.heard(data.decimator(root).cache, Run.discover(root.raw()), data)
    assert torch.equal(Corpus.prior(heard, data), Corpus.load(data, root, CPU).train.log_prior())


def test_sequences(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    heard = Corpus.heard(
        DataConfig().decimator(root).cache, Run.discover(root.raw()), DataConfig(grouping=Grouping.GAP)
    )
    assert Corpus.sequences(heard, Grouping.GAP) == {tokens("the", "a")}
    assert Corpus.sequences(heard.iloc[:0], Grouping.GAP) == frozenset()


def test_load(tmp_path: Path) -> None:
    corpus = Corpus.load(DataConfig(), corpus_root(tmp_path), CPU)
    assert corpus.train.y.tolist() == [THE, IGNORE, A]
    assert set(corpus.validation) == {"Sherlock1"}
    assert corpus.seen == {tokens("the", "zebra", "a")}
    widened = Corpus.load(DataConfig(extra=1), corpus_root(tmp_path / "widened"), CPU)
    assert len(widened.train.y) == 0
    assert set(widened.validation) == {"Sherlock1", "Sherlock+"}
    assert widened.seen == frozenset()
    pooled = Corpus.load(DataConfig(subjects=(0, 3)), corpus_root(tmp_path / "pooled"), CPU)
    assert pooled.train.y.tolist() == [THE, IGNORE, A, THE]
    assert set(pooled.validation) == {"Sherlock1"}

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
from megspeech.data.corpus import METRICS, Corpus, DataConfig
from megspeech.data.grouping import Grouping
from megspeech.data.runs import CHANNELS, Run
from megspeech.data.vocabulary import IGNORE, PRIMARY

SECONDS = 4
CPU = torch.device("cpu")
TRAIN = "sub-0_ses-1_task-Sherlock1_run-1_meg.h5"
HELD = "sub-0_ses-11_task-Sherlock1_run-2_meg.h5"
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
    cache = tmp_path / "processed" / DIRECTORY
    cache.mkdir(parents=True)
    heard = {
        TRAIN: [("the", 0.5, 0, 2.2), ("zebra", 0.9, 0, 2.2), ("a", 1.3, 0, 2.2), ("was", 3.9, 1, 4.9)],
        HELD: [("the", 0.5, 0, 2.0), ("a", 1.0, 0, 2.0)],
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
    return Continuous.load(corpus_root(tmp_path).processed() / DIRECTORY, CPU, names)


def test_trains_on(tmp_path: Path) -> None:
    sherlock = Run(tmp_path / TRAIN)
    timit = Run(tmp_path / "sub-0_ses-1_task-TIMIT_run-1_meg.h5")
    held = Run(tmp_path / HELD)
    only = DataConfig(tasks=("Sherlock1",))
    assert [only.trains_on(r) for r in (sherlock, timit, held)] == [True, False, False]
    assert DataConfig().trains_on(timit)
    assert [DataConfig(refit=True).trains_on(r) for r in (sherlock, held)] == [True, True]


def test_samples() -> None:
    assert DataConfig().samples == RATE
    assert DataConfig(window=2.5).samples == round(2.5 * RATE)


def record(directory: Path, config: dict[str, object]) -> Path:
    directory.mkdir()
    (directory / METRICS).write_text(json.dumps({"config": config}))
    return directory


def test_saved(tmp_path: Path) -> None:
    data = DataConfig(grouping=Grouping.GAP, window=2.0)
    assert DataConfig.saved(record(tmp_path / "nested", {"data": dataclasses.asdict(data)})) == data
    assert DataConfig.saved(record(tmp_path / "flat", {"epochs": 3})) == DataConfig()


def test_window_of(tmp_path: Path) -> None:
    one, two = record(tmp_path / "one", {}), record(tmp_path / "two", {"data": {"window": 2.0}})
    assert DataConfig.window_of([one, one]) == 1.0
    with pytest.raises(ValueError, match=r"windows \[1.0, 2.0\]"):
        DataConfig.window_of([one, two])


def test_words(tmp_path: Path) -> None:
    table = source(tmp_path).words
    sentence = Corpus.words(table, Grouping.SENTENCE)
    assert sentence["word"].tolist() == ["the", "zebra", "a", "the", "a"]
    assert sentence["target"].tolist() == [THE, IGNORE, A, THE, A]
    assert Corpus.words(table, Grouping.GAP)["word"].tolist() == ["the", "a", "the", "a"]


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
    gap = DataConfig(grouping=Grouping.GAP, window=2.0)
    assert Corpus.split(loaded, Corpus.words(loaded.words, Grouping.GAP), gap, np.ones(4)).windows(first).ne(0).all()


def test_domains(tmp_path: Path) -> None:
    loaded = source(tmp_path)
    words = Corpus.words(loaded.words, Grouping.SENTENCE)
    validation = Corpus.domains(loaded, words, [Run(tmp_path / HELD)], DataConfig())
    assert list(validation) == ["Sherlock1"]
    assert validation["Sherlock1"].y.tolist() == [THE, A]


def test_heard(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    heard = Corpus.heard(root.processed() / DIRECTORY, Run.discover(root.raw()), Grouping.SENTENCE)
    assert heard["word"].tolist() == ["the", "zebra", "a"]


def test_sequences(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    heard = Corpus.heard(root.processed() / DIRECTORY, Run.discover(root.raw()), Grouping.GAP)
    assert Corpus.sequences(heard, Grouping.GAP) == {tokens("the", "a")}
    assert Corpus.sequences(heard.iloc[:0], Grouping.GAP) == frozenset()


def test_load(tmp_path: Path) -> None:
    corpus = Corpus.load(DataConfig(), corpus_root(tmp_path), CPU)
    assert corpus.train.y.tolist() == [THE, IGNORE, A]
    assert set(corpus.validation) == {"Sherlock1"}
    assert corpus.seen == {tokens("the", "zebra", "a")}

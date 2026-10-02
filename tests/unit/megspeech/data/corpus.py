from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from core.paths import DataRoot
from megspeech.data.continuous import DIRECTORY, RATE, WORDS, Continuous
from megspeech.data.corpus import Corpus, DataConfig
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
        TRAIN: [("the", 0.5, 0), ("zebra", 0.9, 0), ("a", 1.3, 0), ("was", 3.9, 1)],
        HELD: [("the", 0.5, 0), ("a", 1.0, 0)],
    }
    rows: list[tuple[str, float, str, int]] = []
    for number, (name, words) in enumerate(heard.items()):
        run = Run(tmp_path / name)
        run.meg.touch()
        run.events.parent.mkdir(parents=True, exist_ok=True)
        run.events.write_text("kind")
        signal = np.full((CHANNELS, SECONDS * RATE), number, np.float16)
        signal[:, ::2] += 1
        np.save(cache / f"{run.name}.npy", signal)
        rows += [(run.name, onset, word, sentence) for word, onset, sentence in words]
    pd.DataFrame(rows, columns=["run", "onset", "word", "sentence"]).to_csv(cache / WORDS, sep="\t", index=False)
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


def test_words(tmp_path: Path) -> None:
    table = source(tmp_path).words
    sentence = Corpus.words(table, Grouping.SENTENCE)
    assert sentence["word"].tolist() == ["the", "zebra", "a", "the", "a"]
    assert sentence["target"].tolist() == [THE, IGNORE, A, THE, A]
    assert Corpus.words(table, Grouping.GAP)["word"].tolist() == ["the", "a", "the", "a"]


def test_split(tmp_path: Path) -> None:
    loaded = source(tmp_path)
    words = Corpus.words(loaded.words, Grouping.SENTENCE)
    split = Corpus.split(loaded, words, Grouping.SENTENCE, np.arange(5.0))
    assert split.y.tolist() == [THE, IGNORE, A, THE, A]
    assert split.index.tolist() == [[0, 1, 2] + [-1] * 6, [3, 4] + [-1] * 7]
    assert split.weight.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    window = split.windows(torch.tensor([4]))
    assert window.shape == (1, CHANNELS, RATE)
    assert window.float().abs().tolist() == [[[0.5] * RATE] * CHANNELS]


def test_domains(tmp_path: Path) -> None:
    loaded = source(tmp_path)
    words = Corpus.words(loaded.words, Grouping.SENTENCE)
    validation = Corpus.domains(loaded, words, [Run(tmp_path / HELD)], Grouping.SENTENCE)
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

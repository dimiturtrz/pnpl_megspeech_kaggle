from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.cache.kaggle import Track
from megspeech.data.cache.run_cache import DIRECTORY, WORD_TABLE, DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS, RATE
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import IGNORE, PRIMARY
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import Grouping
from tests.unit.megspeech.data.cache.kaggle import ONSETS, WORD_EPOCHS, kaggle
from tests.unit.megspeech.data.config import HELD, OTHER, TRAIN

SECONDS = 4
CPU = torch.device("cpu")
THE, A = PRIMARY.labels(["the", "a"]).tolist()


def tokens(*words: str) -> tuple[int, ...]:
    return tuple(pd.util.hash_array(np.array(words, dtype=object)).view(np.int64).tolist())


class Root:
    def __init__(self, raw: Path) -> None:
        self.path = raw

    def raw(self) -> Path:
        return self.path

    def processed(self, *parts: str) -> Path:
        return self.path.joinpath("processed", *parts)


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
    pd.DataFrame(rows, columns=["run", "onset", "word", "sentence", "end"]).to_csv(
        cache / WORD_TABLE, sep="\t", index=False
    )
    return cast("DataRoot", Root(tmp_path))


def source(tmp_path: Path) -> DeviceRecording:
    names = [Run(tmp_path / name).name for name in (TRAIN, HELD)]
    return DeviceRecording.load(DataConfig().run_cache(corpus_root(tmp_path)), CPU, names)


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
    assert split.reads().tolist() == [False, False]
    assert split.read is not None
    assert split.read.tolist() == DataConfig.reads(loaded.words["run"].loc[words.index]).tolist()
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
    heard = Corpus.heard(DataConfig().run_cache(root), found, DataConfig())
    assert heard["word"].tolist() == ["the", "zebra", "a"]
    pooled = Corpus.heard(DataConfig().run_cache(root), found, DataConfig(subjects=(0, 3)))
    assert pooled["word"].tolist() == ["the", "zebra", "a"]
    assert pooled["weight"].tolist() == [1.0] * 3


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
    assert Corpus.heard(data.run_cache(root), found, data)["word"].tolist() == words
    assert {r.name for r in data.trained(found)} >= set(Corpus.heard(data.run_cache(root), found, data)["run"])


@pytest.mark.parametrize("data", [DataConfig(), DataConfig(repeat_power=0.5), DataConfig(subjects=(0, 3))])
def test_prior(tmp_path: Path, data: DataConfig) -> None:
    """The prior read from the word table alone is the one training's split counts."""
    root = corpus_root(tmp_path)
    heard = Corpus.heard(data.run_cache(root), Run.discover(root.raw()), data)
    assert torch.equal(Corpus.prior(heard, data), Corpus.load(data, root, CPU).train.log_prior())


def test_sequences(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    heard = Corpus.heard(DataConfig().run_cache(root), Run.discover(root.raw()), DataConfig(grouping=Grouping.GAP))
    assert Corpus.sequences(heard, Grouping.GAP) == {tokens("the", "a")}
    assert Corpus.sequences(heard.iloc[:0], Grouping.GAP) == frozenset()


def test_load(tmp_path: Path) -> None:
    corpus = Corpus.load(DataConfig(), corpus_root(tmp_path), CPU)
    assert corpus.train.y.tolist() == [THE, IGNORE, A]
    assert corpus.positions["position"].tolist() == [0, 1, 2]
    assert set(corpus.positions["run"]) == {Run(tmp_path / TRAIN).name}
    assert set(corpus.validation) == {"Sherlock1"}
    assert corpus.seen == {tokens("the", "zebra", "a")}
    pooled = Corpus.load(DataConfig(subjects=(0, 3)), corpus_root(tmp_path / "pooled"), CPU)
    assert pooled.train.y.tolist() == [THE, IGNORE, A]
    assert set(pooled.validation) == {"Sherlock1"}


def test_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    kaggle(root.raw())
    windows = Corpus.kaggle(root, DataConfig(lead=0.1, baseline=0.5), Track.DEEP, "cpu")
    assert len(windows) == 1 + len(ONSETS) + WORD_EPOCHS
    assert (windows.length, windows.offset, windows.baseline) == (RATE + 5, -5, round(0.5 * RATE))

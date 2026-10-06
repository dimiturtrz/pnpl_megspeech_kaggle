from pathlib import Path

import pytest
import torch

from megspeech.data.config import DataConfig, Refit, Val
from megspeech.data.corpus import Corpus
from megspeech.data.heard import Heard
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import Grouping
from tests.unit.megspeech.data.corpus import CPU, corpus_root, tokens


def test_of(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    found = Run.discover(root.raw())
    heard = Heard.of(DataConfig(val=Val.OFFICIAL).run_cache(root), found, DataConfig(val=Val.OFFICIAL))
    assert heard["word"].tolist() == ["the", "zebra", "a"]
    pooled = Heard.of(DataConfig(val=Val.OFFICIAL).run_cache(root), found, DataConfig(subjects=(0, 3)))
    assert pooled["word"].tolist() == ["the", "zebra", "a"]
    assert pooled["weight"].tolist() == [1.0] * 3


@pytest.mark.parametrize(
    ("data", "words"),
    [
        (DataConfig(val=Val.OFFICIAL, refit=Refit.VAL), ["the", "a", "the", "zebra", "a"]),
        (DataConfig(tasks=("Sherlock2",)), []),
        (DataConfig(folds=2, fold=0), []),
        (DataConfig(folds=2, fold=1), ["the", "zebra", "a"]),
    ],
)
def test_of_trained(tmp_path: Path, data: DataConfig, words: list[str]) -> None:
    """`heard` is the text of exactly the runs training reads, whatever selects them."""
    root = corpus_root(tmp_path)
    found = Run.discover(root.raw())
    assert Heard.of(data.run_cache(root), found, data)["word"].tolist() == words
    assert {r.name for r in data.trained(found)} >= set(Heard.of(data.run_cache(root), found, data)["run"])


@pytest.mark.parametrize(
    "data", [DataConfig(val=Val.OFFICIAL), DataConfig(repeat_power=0.5), DataConfig(subjects=(0, 3))]
)
def test_prior(tmp_path: Path, data: DataConfig) -> None:
    """The prior read from the word table alone is the one training's split counts."""
    root = corpus_root(tmp_path)
    heard = Heard.of(data.run_cache(root), Run.discover(root.raw()), data)
    assert torch.equal(Heard.prior(heard, data), Corpus.load(data, root, CPU).train.log_prior())


def test_sequences(tmp_path: Path) -> None:
    root = corpus_root(tmp_path)
    heard = Heard.of(
        DataConfig(val=Val.OFFICIAL).run_cache(root), Run.discover(root.raw()), DataConfig(grouping=Grouping.GAP)
    )
    assert Heard.sequences(heard, Grouping.GAP) == {tokens("the", "a")}
    assert Heard.sequences(heard.iloc[:0], Grouping.GAP) == frozenset()


def test_under(tmp_path: Path) -> None:
    sequences, prior = Heard.under(corpus_root(tmp_path), DataConfig(val=Val.OFFICIAL, grouping=Grouping.GAP), CPU)
    assert sequences == {tokens("the", "a")}
    assert prior.shape == (VOCAB_SIZE,)

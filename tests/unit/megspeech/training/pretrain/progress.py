from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.model.config import ContextConfig
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.training.pretrain import progress as module
from megspeech.training.pretrain.progress import Probing, Progress

CPU = torch.device("cpu")
SMALL = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
PER_CLASS = 8
ROWS = VOCAB_SIZE * PER_CLASS
DIM = 32
SAMPLES = 128


def split(rows: int = ROWS, ignored: int = 0) -> Split:
    labels = np.arange(rows) % VOCAB_SIZE
    labels[:ignored] = IGNORE
    x = np.random.default_rng(0).standard_normal((rows, CHANNELS, SAMPLES)).astype(np.float32)
    return Split.from_arrays(x, labels, Spans.by_gap(np.arange(rows) * 10.0), CPU)


def encoder() -> SentenceNet:
    torch.manual_seed(0)
    sample = torch.randn(16, CHANNELS, SAMPLES)
    model = SentenceNet(Scaler(sample), context=SMALL, channels=CHANNELS)
    model.whiten.fit(model.scale(sample))
    return model


class Separable(Progress):
    """Embeddings that are each word's class centre plus noise."""

    @override
    def embed(self, net: SentenceNet, split: Split, rows: Int[Tensor, " n"]) -> Float[Tensor, "n e"]:
        generator = torch.Generator().manual_seed(1)
        centres = torch.randn(VOCAB_SIZE, DIM, generator=generator)
        noise = 0.1 * torch.randn(len(rows), DIM)
        return functional.normalize(centres[split.y[rows]] + noise, dim=-1)


class Noise(Progress):
    """Embeddings unrelated to the words."""

    @override
    def embed(self, net: SentenceNet, split: Split, rows: Int[Tensor, " n"]) -> Float[Tensor, "n e"]:
        return functional.normalize(torch.randn(len(rows), DIM), dim=-1)


def progress(kind: type[Progress] = Progress) -> Progress:
    return kind.of(split(), split(), Probing(words=ROWS), CPU)


def test_probing() -> None:
    assert Probing().every == 2
    assert Probing(every=0).every == 0
    with pytest.raises(ValueError, match="probing"):
        Probing(every=-1)
    with pytest.raises(ValueError, match="probing"):
        Probing(words=0)
    with pytest.raises(ValueError, match="probing"):
        Probing(k=0)


def test_of() -> None:
    capped = Progress.of(split(ignored=10), split(), Probing(words=50), CPU)
    assert len(capped.train_rows) == len(capped.val_rows) == 50
    assert (capped.train.y[capped.train_rows] != IGNORE).all()
    assert capped.k == 20


def test_draw() -> None:
    generator = torch.Generator().manual_seed(0)
    drawn = Progress.draw(split(ignored=10), 30, generator)
    assert len(drawn) == 30
    assert torch.equal(drawn, drawn.sort().values)
    assert (split(ignored=10).y[drawn] != IGNORE).all()
    assert len(Progress.draw(split(ignored=10), 10**6, generator)) == ROWS - 10


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[list[str]] = []
    cut: list[DeviceRecording] = []
    built = DeviceRecording(torch.zeros(CHANNELS, 4).half(), np.array([0, 4]), ("a",), pd.DataFrame())
    monkeypatch.setattr(module.Run, "everywhere", classmethod(lambda *_: []))
    monkeypatch.setattr(
        module.DeviceRecording,
        "load",
        classmethod(lambda _c, _cache, _d, names, **_: loaded.append(list(names)) or built),
    )
    monkeypatch.setattr(module.DataConfig, "run_cache", lambda *_: None)
    monkeypatch.setattr(module.DataConfig, "geometry", lambda *_, **__: None)
    monkeypatch.setattr(Progress, "cut", staticmethod(lambda source, _data: cut.append(source) or split(16)))
    out = Progress.load(DataConfig(), DataRoot(tmp_path), built, Probing(words=8, k=5), CPU)
    assert loaded == [[]]
    assert cut == [built, built]
    assert out.k == 5
    assert len(out.train_rows) == 8


def test_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    table = pd.DataFrame({"run": ["a", "a", "b"], "word": ["x", "y", "z"]})
    source = DeviceRecording(torch.zeros(CHANNELS, 4).half(), np.array([0, 2, 4]), ("a", "b"), table)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        module.DataConfig, "carried", lambda _s, words: seen.update(positions=list(words["position"])) or words
    )
    monkeypatch.setattr(
        module.Corpus, "split", staticmethod(lambda _s, words, *_: seen.update(n=len(words)) or split(4))
    )
    assert len(Progress.cut(source, DataConfig()).y) == 4
    assert seen == {"positions": [0, 1, 0], "n": 3}


def test_embed() -> None:
    probe = progress()
    model = encoder()
    out = probe.embed(model, probe.train, torch.arange(10))
    assert out.shape[0] == 10
    assert torch.allclose(out.norm(dim=-1), torch.ones(10), atol=1e-4)


def test_evaluating() -> None:
    model = encoder()
    model.train()
    with Progress.evaluating(model):
        assert not model.training
    assert model.training
    model.eval()
    with Progress.evaluating(model):
        assert not model.training
    assert not model.training


def test_vote() -> None:
    similarity = torch.tensor([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]])
    labels = torch.tensor([1, 1, 2])
    votes = Progress.vote(similarity, labels, 2, 0.5)
    assert votes.shape == (2, VOCAB_SIZE)
    assert votes[0].argmax() == 1
    assert votes[0, 2] == 0
    assert torch.isclose(votes[0, 1], torch.tensor([0.9 / 0.5, 0.8 / 0.5]).exp().sum() / 2)
    assert votes[1].argmax() == 2
    assert Progress.vote(similarity, labels, 99, 0.5).shape == (2, VOCAB_SIZE)
    frequent = torch.tensor([[0.5, 0.5, 0.5, 0.6]])
    assert Progress.vote(frequent, torch.tensor([1, 1, 1, 2]), 4, 1.0)[0].argmax() == 2


def test_score() -> None:
    assert progress(Separable).score(encoder()) > 0.9
    assert 0.0 <= progress(Noise).score(encoder()) < 0.4
    assert 0.0 <= progress().score(encoder()) <= 1.0


def test_read() -> None:
    read = progress(Separable).read(encoder(), encoder())
    assert set(read) == {"probe/student", "probe/teacher"}
    assert all(v > 0.9 for v in read.values())

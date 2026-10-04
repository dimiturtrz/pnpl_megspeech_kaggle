import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import DENSE, FEATURES, TRAIN, Batch, Features, SpanText, Split
from megspeech.data.batching.windows import Windows
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD, Spans

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


class Kaggle:
    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": [0, 1, 2],
                "subject": 0,
                "source": ["sentence", "sentence", "word"],
                "epoch": [0, 0, 0],
                "onset_s": [0.0, 0.3, 0.0],
            }
        )


def test_from_arrays() -> None:
    assert len(split().windows) == 12
    assert split().windows(torch.tensor([0])).shape == (1, CHANNELS, SAMPLES)


def test_of() -> None:
    spans = Spans.by_gap(np.array([0.0, 0.1, 9.0]))
    at = torch.tensor([0, 10, 20])
    windows = Windows(torch.zeros(CHANNELS, 100), at, at + 50)
    held = Split.of(windows, (np.array([0, 1, 2]),) * 2, spans, np.array([1.0, 1.0, 3.0]))
    assert held.y.tolist() == [0, 1, 2]
    assert held.index.shape[0] == 2
    assert held.weight.tolist() == [1.0, 1.0, 3.0]


def test_sample() -> None:
    assert split().sample(5).shape == (5, CHANNELS, SAMPLES)
    assert split().sample(100).shape == (12, CHANNELS, SAMPLES)


def test_labelled() -> None:
    held = dataclasses.replace(split(), y=torch.tensor([0, IGNORE, 2]))
    assert held.labelled().tolist() == [0, 2]


def test_lone() -> None:
    """The dense words share a span; each word 2 s after its neighbour is a span of its own."""
    assert split().lone().tolist() == [False] * DENSE + [True] * (12 - DENSE)


def test_log_prior() -> None:
    assert torch.allclose(split().log_prior().exp().sum(), torch.tensor(1.0))
    heavy = dataclasses.replace(split(), weight=(torch.arange(12) % 3 == 0).float() * 100)
    assert heavy.log_prior().argmax() == 0


def test_prior() -> None:
    prior = Split.prior(torch.tensor([0, 0, 1, IGNORE, IGNORE, IGNORE]), torch.ones(6))
    assert prior.argmax() == 0
    assert torch.allclose(prior.exp().sum(), torch.tensor(1.0))


def test_split_novel() -> None:
    held = split()
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(12 - DENSE) * 2.0]
    everything = frozenset(SpanText(Spans.by_gap(onset).index, held.y.numpy()).sequences())
    assert len(held.novel(everything)) == 0
    assert held.novel(frozenset()).tolist() == list(range(DENSE))
    unknown = dataclasses.replace(held, y=torch.where(torch.arange(12) == 1, IGNORE, held.y))
    assert unknown.novel(frozenset()).tolist() == [0, 2, 3]
    retold = dataclasses.replace(held, token=held.token + 100)
    assert retold.novel(everything).tolist() == list(range(DENSE))


def test_text() -> None:
    held = split()
    assert held.text.index.tolist() == held.index.tolist()
    assert held.text.token.tolist() == held.token.tolist()


def test_recurring() -> None:
    onset = np.array([0, 0.1, 0.2, 9, 9.1, 9.2, 20, 20.1, 30, 30.1])
    y = np.array([0, 1, 2, 0, 1, 2, 3, 4, 3, 4])
    twice = Split.from_arrays(np.zeros((10, CHANNELS, SAMPLES), np.float32), y, Spans.by_gap(onset), CPU)
    assert twice.recurring(1.0).weight.tolist() == [0.5] * 6 + [1.0] * 4
    assert twice.recurring(0.0).weight.tolist() == [1.0] * 10


def test_batch() -> None:
    batch = split().batch(torch.tensor([1, 0]), rows=6)
    assert batch.slot[:, :DENSE].tolist() == [[0, 6, 6, 6], [1, 2, 3, 4]]
    assert torch.equal(batch.x[:5], split().windows(torch.tensor([DENSE, 0, 1, 2, 3])))
    assert batch.onset.shape == batch.slot.shape
    overflow = split().batch(torch.tensor([0]), rows=2)
    assert overflow.slot[:, :DENSE].tolist() == [[0, 1, 2, 2]]
    assert overflow.index[:, :DENSE].tolist() == [[0, 1, PAD, PAD]]
    jittered = split().batch(torch.tensor([1, 0]), rows=6, jitter=1)
    assert jittered.x.shape == batch.x.shape
    assert torch.equal(jittered.slot, batch.slot)


def test_capacity() -> None:
    s = split()
    natural = s.weight[s.index[:, 0]]
    assert s.capacity(1, natural) == 8
    assert s.capacity(100, natural) == 192
    longest = (s.index != PAD).sum(dim=1) == (s.index != PAD).sum(dim=1).max()
    assert s.capacity(100, longest.float()) > s.capacity(100, natural)


def test_kaggle_like() -> None:
    assert split().kaggle_like().tolist() == list(range(DENSE))
    unknown = dataclasses.replace(split(), y=torch.where(torch.arange(12) == 0, IGNORE, split().y))
    assert unknown.kaggle_like().tolist() == list(range(1, DENSE))


def test_from_kaggle() -> None:
    windows = Windows.stacked(torch.zeros(3, CHANNELS, SAMPLES))
    held = Split.from_kaggle(Kaggle(), windows)
    assert held.windows is windows
    assert held.index[:, :2].tolist() == [[0, 1], [2, -1]]
    assert held.onset[0, :2].tolist() == pytest.approx([0.0, 0.3])
    with pytest.raises(ValueError, match="2 Kaggle test windows for 3"):
        Split.from_kaggle(Kaggle(), Windows.stacked(torch.zeros(2, CHANNELS, SAMPLES)))


def test_target() -> None:
    slot = torch.tensor([[0, 1, 2]])
    batch = Batch(torch.zeros(2, 1, 1), torch.zeros(1, 3), slot, torch.tensor([[4, 3, PAD]]), torch.zeros(2))
    assert batch.target(torch.tensor([0, 1, 2, 7, 8])).tolist() == [[8, 7, IGNORE]]


TOKEN = np.array([5, 6, 7, 5, 6, 7, 9, 9])
INDEX = np.array([[0, 1, 2], [3, 4, 5], [6, 7, PAD]])


def test_sequences() -> None:
    assert SpanText(INDEX, TOKEN).sequences() == [(5, 6, 7), (5, 6, 7), (9, 9)]
    spans = Spans.by_gap(np.array([0.0, 0.2, 9.0]))
    assert set(SpanText(spans.index, np.array([4, 5, 6])).sequences()) == {(4, 5), (6,)}


def test_repeats() -> None:
    """A recited text heard twice weighs each of its words half at power 1; a span too short to recite keeps 1."""
    text = SpanText(INDEX, TOKEN)
    assert text.repeats(1.0).tolist() == [0.5] * 6 + [1.0, 1.0]
    assert text.repeats(0.0).tolist() == [1.0] * 8
    shifted = SpanText(np.array([[PAD, 0, 1, 2], [3, 4, 5, PAD]]), TOKEN[:6])
    assert shifted.repeats(1.0).tolist() == [0.5] * 6
    unspanned = SpanText(np.array([[0, 1, 2]]), TOKEN)
    assert unspanned.repeats(1.0).tolist() == [1.0] * 8


def test_span_text_novel() -> None:
    text = SpanText(INDEX, TOKEN)
    assert text.novel(frozenset()).tolist() == list(range(8))
    assert text.novel(frozenset({(5, 6, 7)})).tolist() == [6, 7]
    assert len(SpanText(np.array([[0, PAD]]), TOKEN).novel(frozenset())) == 0


FEATURE = torch.arange(6.0).reshape(3, 2)


def saved(tmp_path: Path) -> Features:
    features = Features(tmp_path / "f")
    features.save(TRAIN, FEATURE)
    return features


def windows(count: int = 3) -> Windows:
    return Windows.stacked(torch.ones(count, 4, 5))


def test_under(tmp_path: Path):
    root = DataRoot(tmp_path)
    assert Features.under(root, "x").directory == root.kind("models") / FEATURES / "x"
    assert Features.under(root, "").directory is None


def test_path(tmp_path: Path):
    assert Features(tmp_path).path(TRAIN) == tmp_path / "train.pt"
    with pytest.raises(ValueError, match="no features"):
        Features().path(TRAIN)


def test_save(tmp_path: Path):
    path = saved(tmp_path).path(TRAIN)
    assert torch.equal(torch.load(path, weights_only=True), FEATURE)


def test_read(tmp_path: Path):
    raw = windows()
    assert Features().read(TRAIN, raw) is raw
    read = saved(tmp_path).read(TRAIN, raw)
    assert torch.equal(read(torch.tensor([2, 0])), FEATURE[[2, 0], :, None])
    resolved = Features(tmp_path / "r")
    resolved.save(TRAIN, FEATURE.reshape(3, 1, 2))
    assert torch.equal(resolved.read(TRAIN, raw)(torch.tensor([1])), FEATURE[[1], None])
    with pytest.raises(ValueError, match="2 windows"):
        saved(tmp_path).read(TRAIN, windows(2))


def test_split(tmp_path: Path):
    split = Split.of(windows(), (np.zeros(3, np.int64),) * 2, Spans.by_gap(np.array([0.0, 0.1, 9.0])), np.ones(3))
    read = saved(tmp_path).split(TRAIN, split)
    assert read.windows.length == 1
    assert read.y is split.y

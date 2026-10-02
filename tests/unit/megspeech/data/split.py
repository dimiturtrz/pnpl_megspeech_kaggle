import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.runs import CHANNELS
from megspeech.data.spans import PAD, Spans
from megspeech.data.split import DENSE, Split
from megspeech.data.vocabulary import IGNORE
from megspeech.data.windows import Windows

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


class Holdout:
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


def test_log_prior() -> None:
    assert torch.allclose(split().log_prior().exp().sum(), torch.tensor(1.0))
    heavy = dataclasses.replace(split(), weight=(torch.arange(12) % 3 == 0).float() * 100)
    assert heavy.log_prior().argmax() == 0


def test_prior() -> None:
    prior = Split.prior(torch.tensor([0, 0, 1, IGNORE, IGNORE, IGNORE]), torch.ones(6))
    assert prior.argmax() == 0
    assert torch.allclose(prior.exp().sum(), torch.tensor(1.0))


def test_seen() -> None:
    seen = Split.seen([(np.array([4, 5, 6]), Spans.by_gap(np.array([0.0, 0.2, 9.0])))])
    assert seen == {(4, 5), (6,)}


def test_novel() -> None:
    held = split()
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(12 - DENSE) * 2.0]
    everything = Split.seen([(held.y.numpy(), Spans.by_gap(onset))])
    assert len(held.novel(everything)) == 0
    assert held.novel(frozenset()).tolist() == list(range(DENSE))
    unknown = dataclasses.replace(held, y=torch.where(torch.arange(12) == 1, IGNORE, held.y))
    assert unknown.novel(frozenset()).tolist() == [0, 2, 3]
    retold = dataclasses.replace(held, token=held.token + 100)
    assert retold.novel(everything).tolist() == list(range(DENSE))


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


def test_target() -> None:
    data = split()
    assert data.batch(torch.tensor([0]), rows=2).target(data.y)[:, :DENSE].tolist() == [[0, 1, IGNORE, IGNORE]]


def test_capacity() -> None:
    assert split().capacity(1) == 8
    assert split().capacity(100) == 192


def test_dense() -> None:
    assert split().dense().tolist() == list(range(DENSE))
    unknown = dataclasses.replace(split(), y=torch.where(torch.arange(12) == 0, IGNORE, split().y))
    assert unknown.dense().tolist() == list(range(1, DENSE))


def test_from_holdout() -> None:
    windows = Windows.stacked(torch.zeros(3, CHANNELS, SAMPLES))
    held = Split.from_holdout(Holdout(), windows)
    assert held.windows is windows
    assert held.index[:, :2].tolist() == [[0, 1], [2, -1]]
    assert held.onset[0, :2].tolist() == pytest.approx([0.0, 0.3])
    with pytest.raises(ValueError, match="2 holdout windows for 3"):
        Split.from_holdout(Holdout(), Windows.stacked(torch.zeros(2, CHANNELS, SAMPLES)))

import dataclasses
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from megspeech.epochs import Epochs
from megspeech.spans import Spans
from megspeech.train import ADJUSTMENTS, DENSE, Head, Score, Split, TrainConfig, Trainer
from megspeech.wordnet import CHANNELS, WORDS

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


class Constant(nn.Module):
    @override
    def forward(self, x: torch.Tensor, onset: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(*mask.shape, WORDS)
        logits[..., 0] = 1
        return logits, logits


def test_from_arrays() -> None:
    assert split().x.shape == (12, CHANNELS, SAMPLES)


def test_gather() -> None:
    epochs = Epochs(np.zeros((2, CHANNELS, SAMPLES), np.float32), np.array([0, 1]), np.array([0.0, 0.1]))
    gathered = Split.gather([(epochs, 1.0), (epochs, 3.0)], CPU)
    assert len(gathered.y) == 4
    assert gathered.index.shape[0] == 2
    assert gathered.weight.tolist() == [1.0, 1.0, 3.0, 3.0]


def test_on_device() -> None:
    spans = Spans.by_gap(np.array([0.0, 0.1, 9.0]))
    held = Split.on_device(torch.zeros(3, CHANNELS, SAMPLES), np.array([0, 1, 2]), spans, np.ones(3))
    assert held.y.tolist() == [0, 1, 2]
    assert held.index.shape[0] == 2


def test_log_prior() -> None:
    assert torch.allclose(split().log_prior().exp().sum(), torch.tensor(1.0))
    heavy = dataclasses.replace(split(), weight=(torch.arange(12) % 3 == 0).float() * 100)
    assert heavy.log_prior().argmax() == 0


def test_batch() -> None:
    x, onset, mask, words = split().batch(torch.tensor([0]))
    assert len(x) == mask.sum() == DENSE
    assert words.tolist() == list(range(DENSE))
    assert onset.shape == mask.shape


def test_dense() -> None:
    assert split().dense().tolist() == list(range(DENSE))


def test_logits() -> None:
    out = Trainer(TrainConfig(), CPU).logits(Constant(), split())
    assert out[Head.CONTEXT].shape == (12, WORDS)
    assert torch.all(out[Head.WORD].argmax(dim=1) == 0)


def test_evaluate() -> None:
    logits = torch.zeros(3, WORDS)
    logits[:, 0] = 1
    result = Trainer(TrainConfig(), CPU).evaluate(logits, torch.tensor([0, 0, 0]), torch.zeros(WORDS))
    assert result.bacc10 == pytest.approx(1.0)
    assert result.n == 3


def test_best() -> None:
    score = Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7)
    assert (score.tau, score.bacc10, score.n) == (0.5, 0.4, 7)


def test_assess() -> None:
    report = Trainer(TrainConfig(), CPU).assess(Constant(), split(), torch.zeros(WORDS))
    assert report.dense.n == DENSE
    assert report.context.n == len(split().y)


def test_report() -> None:
    domains = {"a": split(), "b": split(n=8, seed=1)}
    report = Trainer(TrainConfig(), CPU).report(Constant(), domains, torch.zeros(WORDS), 3, 0.5)
    assert report.epoch == 3
    assert set(report.domains) == {"a", "b"}
    assert report.macro.n == 12 + 8
    mean = (report.domains["a"].context.by_tau[0.0] + report.domains["b"].context.by_tau[0.0]) / 2
    assert report.macro.by_tau[ADJUSTMENTS[0]] == pytest.approx(mean)


def test_fit() -> None:
    model, history = Trainer(TrainConfig(epochs=1, batch=4), CPU).fit(split(), {"a": split(seed=1)}, torch.zeros(WORDS))
    assert len(history) == 1
    assert isinstance(model, nn.Module)


def test_from_holdout() -> None:
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

        def iter_windows(self, batch_size: int) -> list[tuple[np.ndarray, None]]:
            return [(np.zeros((3, CHANNELS, 250)), None)]

    held = Split.from_holdout(Holdout(), CPU)
    assert held.x.shape == (3, CHANNELS, SAMPLES)
    assert held.index[:, :2].tolist() == [[0, 1], [2, -1]]

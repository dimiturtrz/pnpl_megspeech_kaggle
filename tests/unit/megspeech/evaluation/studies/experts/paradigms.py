import math

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.constants import RATE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.studies.experts.boosted import Rows
from megspeech.evaluation.studies.experts.paradigms import (
    BANDS,
    COVARIANCE_COMPONENTS,
    SEGMENTS,
    Extractor,
    Features,
    Paradigm,
    Spatial,
)

CPU = torch.device("cpu")
WORDS = 6
CHANNELS = 16
COMPONENTS = 13
ALPHA = 10.0
LEVEL = 3.0


def split() -> Split:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(WORDS, CHANNELS, RATE)).astype(np.float32) + LEVEL
    return Split.from_arrays(x, np.arange(WORDS) % 2, Spans.by_gap(np.arange(WORDS) * 2.0), CPU)


def spatial() -> Spatial:
    return Spatial(torch.eye(CHANNELS)[:, :COMPONENTS])


def test_fitted() -> None:
    x = np.zeros((WORDS, CHANNELS, RATE), np.float32)
    x[:, 3] = np.random.default_rng(0).normal(size=(WORDS, RATE)) * 10
    data = Split.from_arrays(x, np.zeros(WORDS, np.int64), Spans.by_gap(np.arange(WORDS) * 2.0), CPU)
    basis = Spatial.fitted([data], 1).basis
    assert basis.shape == (CHANNELS, 1)
    assert basis[3, 0].abs().item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no windows"):
        Spatial.fitted([], 1)


def test_call() -> None:
    x = torch.randn(2, CHANNELS, RATE)
    assert torch.equal(spatial()(x), x[:, :COMPONENTS])


def test_band() -> None:
    t = torch.arange(RATE) / RATE
    x = torch.sin(2 * math.pi * ALPHA * t).expand(1, 1, RATE)
    power = Extractor.band(x)
    alpha = next(i for i, (low, high) in enumerate(BANDS) if low <= ALPHA < high)
    assert power.shape == (1, len(BANDS))
    assert int(power.argmax()) == alpha


def test_means() -> None:
    x = torch.arange(RATE, dtype=torch.float32).expand(2, 3, RATE)
    means = Extractor.means(x)
    assert means.shape == (2, 3 * SEGMENTS)
    assert means[0, 0].item() == pytest.approx((RATE // SEGMENTS - 1) / 2)


def test_covariance() -> None:
    x = torch.randn(4, COMPONENTS, RATE * 40)
    tangent = Extractor.covariance(x)
    assert tangent.shape == (4, COVARIANCE_COMPONENTS * (COVARIANCE_COMPONENTS + 1) // 2)
    assert tangent.abs().max().item() < 1.0


def test_extractor() -> None:
    features = Extractor(spatial())(split())
    assert set(features) == set(Paradigm)
    assert all(len(part) == WORDS for part in features.values())
    assert features[Paradigm.LEVEL].mean().item() == pytest.approx(LEVEL, abs=0.2)
    assert features[Paradigm.MEANS].mean().abs().item() < 0.1


def test_joined() -> None:
    part = {paradigm: torch.ones(2, 3) for paradigm in Paradigm}
    joined = Features.joined([part, part])
    assert all(value.shape == (4, 3) for value in joined.values())


def test_designs() -> None:
    def rows(count: int) -> Rows:
        return Rows(
            torch.zeros(count, 4),
            torch.zeros(count, 2),
            torch.zeros(count, 4),
            torch.zeros(count, 2),
            torch.zeros(count, dtype=torch.long),
            torch.full((count, 2), -1),
            torch.zeros(count, dtype=torch.long),
        )

    def features(count: int) -> dict[Paradigm, torch.Tensor]:
        return {paradigm: torch.ones(count, 5) for paradigm in Paradigm}

    train, val = Features(features(3), features(2)).designs(rows(3), rows(2), (Paradigm.BAND, Paradigm.LEVEL))
    assert train.x.shape == (3, 4 + 10)
    assert val.x.shape == (2, 4 + 10)
    assert torch.equal(val.base, rows(2).base)

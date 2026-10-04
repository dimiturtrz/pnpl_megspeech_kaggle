import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.training.sampling.curriculum import Annealed, Natural, Robust


def split(subject: list[int], weight: list[float]) -> Split:
    n = len(subject)
    spans = Spans(np.arange(n)[:, None], np.zeros((n, 1)))
    windows = Windows.stacked(torch.zeros(n, 1, 2))
    labels = np.zeros(n, dtype=np.int64)
    return Split.of(windows, (labels, labels), spans, np.array(weight), np.array(subject))


TRAIN = split([0, 0, 1, 2], [1.0, 3.0, 2.0, 5.0])


def test_natural_weights() -> None:
    assert Natural().weights(TRAIN, 0, 10).tolist() == [1.0, 3.0, 2.0, 5.0]


def test_annealed_of() -> None:
    assert isinstance(Annealed.of(0.0), Natural)
    assert Annealed.of(0.9) == Annealed(0.9)


@pytest.mark.parametrize(("epoch", "deep"), [(0, 1 / 3), (9, 0.9), (3, 1 / 3 + (0.9 - 1 / 3) / 3)])
def test_annealed_weights(epoch: int, deep: float) -> None:
    """Subject shares follow the schedule; inside the Deep subject spans keep their 1:3 ratio."""
    weights = Annealed(0.9).weights(TRAIN, epoch, 10)
    shares = weights / weights.sum()
    assert float(shares[:2].sum()) == pytest.approx(deep)
    assert float(shares[2]) == pytest.approx((1 - deep) / 2)
    assert float(shares[1] / shares[0]) == pytest.approx(3.0)


def test_annealed_weights_single_subject() -> None:
    lone = split([0, 0], [1.0, 3.0])
    weights = Annealed(0.9).weights(lone, 5, 10)
    assert torch.allclose(weights / weights.sum(), torch.tensor([0.25, 0.75]))


SPANNED = Split.of(
    Windows.stacked(torch.zeros(6, 1, 2)),
    (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)),
    Spans(np.array([[0, 1, 2], [3, -1, -1], [4, 5, -1]]), np.array([[0, 0.2, 0.4], [0, 0, 0], [0, 1.0, 0]])),
    np.ones(6),
)
"""Spans of 3 words at 5 words/s, 1 word (rate 0), 2 words at 1 word/s: each the top, bottom and middle tercile."""


def robust(strata: list[int], power: float = 1.0) -> Robust:
    zeros = torch.zeros(max(strata) + 1)
    return Robust(Natural(), torch.tensor(strata), power, torch.ones_like(zeros), zeros.clone(), zeros.clone())


def test_robust_of() -> None:
    assert isinstance(Robust.of(Natural(), SPANNED, 0.0), Natural)
    made = Robust.of(Natural(), SPANNED, 2.0)
    assert isinstance(made, Robust)
    assert made.strata.tolist() == [8, 0, 4]
    assert made.tilt.tolist() == [1.0] * 9


def test_stratify() -> None:
    """Subject, then length tercile, then rate tercile; a second subject's strata sit past the first's."""
    assert Robust.stratify(SPANNED).tolist() == [8, 0, 4]
    other = replace(SPANNED, subject=torch.ones(6, dtype=torch.int64))
    assert Robust.stratify(other).tolist() == [17, 9, 13]


def test_binned() -> None:
    assert Robust.binned(torch.tensor([3.0, 1.0, 2.0])).tolist() == [2, 0, 1]
    assert Robust.binned(torch.arange(6.0)).tolist() == [0, 0, 1, 1, 2, 2]


def test_observe() -> None:
    """Each labelled word's cross-entropy lands in its span's stratum; ignored positions add neither loss nor count."""
    Natural().observe(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    Annealed(0.9).observe(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    made = robust([1, 0, 1])
    target = torch.tensor([[0, 1, IGNORE], [3, IGNORE, IGNORE]])
    made.observe(torch.tensor([0, 2]), torch.zeros(2, 3, 50), target)
    assert made.counts.tolist() == [0.0, 3.0]
    assert made.sums.tolist() == pytest.approx([0.0, 3 * math.log(50)])


def test_robust_weights() -> None:
    """A stratum above the mean loss is drawn more by its ratio to the power, one below never less; the losses are
    spent, and an epoch observing nothing keeps the tilt."""
    made = robust([0, 1, 0, 1], power=2.0)
    made.sums += torch.tensor([4.0, 1.0])
    made.counts += torch.tensor([1.0, 1.0])
    expected = [1.6**2, 3.0, 2 * 1.6**2, 5.0]
    assert made.weights(TRAIN, 1, 10).tolist() == pytest.approx(expected)
    assert made.counts.tolist() == [0.0, 0.0]
    assert made.weights(TRAIN, 2, 10).tolist() == pytest.approx(expected)

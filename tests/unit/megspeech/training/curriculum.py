import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.spans.spans import Spans
from megspeech.training.curriculum import Annealed, Natural


def split(subject: list[int], weight: list[float]) -> Split:
    n = len(subject)
    spans = Spans(np.arange(n)[:, None], np.zeros((n, 1)))
    windows = Windows.stacked(torch.zeros(n, 1, 2))
    labels = np.zeros(n, dtype=np.int64)
    return Split.of(windows, (labels, labels), spans, np.array(weight), np.array(subject))


TRAIN = split([0, 0, 1, 2], [1.0, 3.0, 2.0, 5.0])


def test_natural_weights() -> None:
    assert Natural().weights(TRAIN, 0, 10).tolist() == [1.0, 3.0, 2.0, 5.0]


def test_of() -> None:
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

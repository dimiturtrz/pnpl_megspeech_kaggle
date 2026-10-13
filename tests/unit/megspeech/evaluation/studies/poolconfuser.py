import pytest
import torch

from megspeech.evaluation.studies.poolconfuser import Confusers

SPAN = torch.tensor([[0, 0, 0, 0, 1, 1]])
PLACE = torch.tensor([[0, 1, 2, 3, 0, 1]])
LABEL = torch.tensor([[0, 0, 1, 0, 0, 2]])


def edges(*pairs: tuple[int, int]) -> torch.Tensor:
    keep = torch.zeros(1, 6, 6, dtype=torch.bool)
    for i, j in pairs:
        keep[0, i, j] = True
    return keep


def test_of() -> None:
    found = Confusers.of(edges((0, 1), (0, 3), (0, 4), (4, 5)), SPAN, PLACE, LABEL)
    assert found.same_span == pytest.approx(3 / 4)
    assert found.within_1 == pytest.approx(2 / 4)
    assert found.within_2 == pytest.approx(2 / 4)
    assert found.purity_all == pytest.approx(3 / 4)
    assert found.purity_same_span == pytest.approx(2 / 3)
    assert found.purity_within_2 == pytest.approx(0.5)
    assert found.purity_outside == pytest.approx(1.0)
    assert found.chance_span == pytest.approx((4 * 3 + 2 * 1) / (6 * 5))


def test_mean() -> None:
    a = Confusers.of(edges((0, 1)), SPAN, PLACE, LABEL)
    b = Confusers.of(edges((0, 2)), SPAN, PLACE, LABEL)
    assert Confusers.mean([a, b]).within_1 == pytest.approx(0.5)
    assert Confusers.mean([a, b]).purity_all == pytest.approx(0.5)

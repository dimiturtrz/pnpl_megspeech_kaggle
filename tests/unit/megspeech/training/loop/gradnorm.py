import pytest
import torch
from torch import nn

from megspeech.training.loop.gradnorm import GradNorms


def graded(first: float, second: float) -> tuple[nn.Parameter, nn.Parameter]:
    a, b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    a.grad, b.grad = torch.tensor([first]), torch.tensor([second])
    return a, b


def test_record() -> None:
    inside, _ = graded(3.0, 4.0)
    norms = GradNorms((inside,))
    norms.record(torch.tensor(5.0))
    assert [row.tolist() for row in norms.rows] == [[5.0, 3.0]]


def test_record_without_encoder_gradients() -> None:
    frozen = nn.Parameter(torch.zeros(1))
    norms = GradNorms((frozen,))
    norms.record(torch.tensor(2.0))
    assert norms.rows[0].tolist() == [2.0, 0.0]


def test_drained() -> None:
    inside, _ = graded(3.0, 4.0)
    norms = GradNorms((inside,))
    assert norms.drained() == {}
    norms.record(torch.tensor(5.0))
    inside.grad = torch.tensor([1.0])
    norms.record(torch.tensor(1.0))
    assert norms.drained() == {
        "total_mean": pytest.approx(3.0),
        "total_max": 5.0,
        "encoder_mean": pytest.approx(2.0),
        "encoder_max": 3.0,
    }
    assert norms.rows == []

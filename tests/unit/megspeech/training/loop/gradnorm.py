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


def test_record_matches_per_parameter_norms() -> None:
    torch.manual_seed(0)
    shapes = [(64, 32, 3), (64,), (128, 64), (7,), (32, 32, 5)] * 12 + [(3,), (5, 5)]
    params = tuple(nn.Parameter(torch.zeros(s)) for s in shapes)
    for i, p in enumerate(params):
        p.grad = None if i % 9 == 4 else torch.randn(p.shape) * 10.0 ** (i % 5 - 2)
    norms = GradNorms(params)
    norms.record(torch.tensor(1.0))
    total = torch.zeros(())
    per_parameter = [p.grad.norm() for p in params if p.grad is not None]
    expected = torch.linalg.vector_norm(torch.stack([*per_parameter, total]))
    torch.testing.assert_close(norms.rows[0][1], expected, rtol=1e-6, atol=0.0)

import math

import pytest
import torch

from megspeech.objective.heads import BIAS, DIM, TEMPERATURE
from megspeech.objective.semantic import Semantic

WIDTH = 6
GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    assert Semantic(WIDTH).forward(torch.randn(2, 3, WIDTH, generator=GENERATOR)).shape == (2, 3, DIM)


def softplus(x: float) -> float:
    return math.log1p(math.exp(x))


def test_loss() -> None:
    """Orthogonal targets, rows on their own: each row's own kind has cosine 1, the other kinds 0."""
    semantic = Semantic(WIDTH)
    eye = torch.eye(DIM)[:3]
    targets = eye[[0, 1, 0]]
    positive, negative = math.exp(TEMPERATURE) + BIAS, BIAS
    pair = softplus(-positive) + softplus(negative)
    loss = semantic.loss(targets, targets, torch.tensor([4, 7, 4]))
    assert loss.item() == pytest.approx(pair, rel=1e-5)
    padded = semantic.loss(torch.cat([targets, eye[2:]]), torch.cat([targets, eye[2:]]), torch.tensor([4, 7, 4, -1]))
    assert padded.item() == pytest.approx(pair, rel=1e-5)
    assert semantic.loss(targets, targets, torch.full((3,), -100)).item() == 0.0

import pytest
import torch

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.context import CosineHead, Objective
from megspeech.objective.scorer import DIM

GENERATOR = torch.Generator().manual_seed(0)
WIDTH = 8


def test_cosine_head_forward() -> None:
    head = CosineHead(WIDTH, torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR))
    logits = head.forward(torch.randn(2, 3, WIDTH, generator=GENERATOR))
    assert logits.shape == (2, 3, VOCAB_SIZE)
    assert logits.abs().max() <= head.temperature.exp() - head.bias + 1e-4


@pytest.mark.parametrize("objective", list(Objective))
def test_head(objective: Objective) -> None:
    assert objective.head(WIDTH, None)(torch.zeros(1, WIDTH)).shape == (1, VOCAB_SIZE)


@pytest.mark.parametrize("objective", list(Objective))
def test_loss(objective: Objective) -> None:
    y = torch.tensor([3, 7])
    right = torch.full((2, VOCAB_SIZE), -10.0)
    right[torch.arange(2), y] = 10
    assert objective.loss(right, y, 0.0) < objective.loss(-right, y, 0.0)
    padded = torch.cat([right, torch.randn(1, VOCAB_SIZE)])
    assert torch.allclose(objective.loss(padded, torch.tensor([3, 7, IGNORE]), 0.1), objective.loss(right, y, 0.1))
    ones = torch.ones(VOCAB_SIZE)
    assert torch.allclose(objective.loss(right, y, 0.1, ones), objective.loss(right, y, 0.1))

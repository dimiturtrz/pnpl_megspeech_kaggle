import pytest
import torch

from megspeech.data.vocabulary import IGNORE, WORDS
from megspeech.objective.heads import DIM, CosineHead, Objective

GENERATOR = torch.Generator().manual_seed(0)
WIDTH = 8


def test_cosine_head_forward() -> None:
    head = CosineHead(WIDTH, torch.randn(WORDS, DIM, generator=GENERATOR))
    logits = head.forward(torch.randn(2, 3, WIDTH, generator=GENERATOR))
    assert logits.shape == (2, 3, WORDS)
    assert logits.abs().max() <= head.temperature.exp() - head.bias + 1e-4


@pytest.mark.parametrize("objective", list(Objective))
def test_head(objective: Objective) -> None:
    assert objective.head(WIDTH, None)(torch.zeros(1, WIDTH)).shape == (1, WORDS)


@pytest.mark.parametrize("objective", list(Objective))
def test_loss(objective: Objective) -> None:
    y = torch.tensor([3, 7])
    right = torch.full((2, WORDS), -10.0)
    right[torch.arange(2), y] = 10
    assert objective.loss(right, y, 0.0) < objective.loss(-right, y, 0.0)
    padded = torch.cat([right, torch.randn(1, WORDS)])
    assert torch.allclose(objective.loss(padded, torch.tensor([3, 7, IGNORE]), 0.1), objective.loss(right, y, 0.1))
    ones = torch.ones(WORDS)
    assert torch.allclose(objective.loss(right, y, 0.1, ones), objective.loss(right, y, 0.1))

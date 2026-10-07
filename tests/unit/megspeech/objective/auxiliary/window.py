from typing import cast

import pytest
import torch

from megspeech.objective.auxiliary.window import OFFSETS, Window, WindowTerm
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.t5 import Lexicon

GENERATOR = torch.Generator().manual_seed(0)
ROWS = torch.ones(2, 4)
TOKENS = torch.tensor([[10, 20, 10, -100], [20, -100, -100, -100]])


def term() -> WindowTerm:
    return WindowTerm(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def test_forward() -> None:
    assert Window(6).forward(torch.randn(2, 3, 6, generator=GENERATOR)).shape == (2, 3, OFFSETS, DIM)


def test_lag_weights() -> None:
    """Offsets 0, 1, 2 hold their whole 0.2-0.5 s response, -1 and +3 a sixth to five sixths of it."""
    weights = WindowTerm.lag_weights()
    assert weights.sum().item() == pytest.approx(1.0)
    held = weights / weights[1]
    assert held.tolist() == pytest.approx([5 / 6, 1.0, 1.0, 1.0, 1 / 6])


def test_head() -> None:
    assert isinstance(term().head(16), Window)


def test_weighted() -> None:
    """Positive, differentiable into the projection, 0 when no token is in the lexicon."""
    windowed = term()
    head = windowed.head(16)
    output = head(torch.randn(2, 4, 16, generator=GENERATOR))
    value = windowed.weighted(head, output, TOKENS, ROWS)
    assert value.ndim == 0
    assert value.item() > 0
    value.backward()
    assert cast("Window", head).project.weight.grad is not None
    assert windowed.weighted(head, output, torch.full((2, 4), 99), ROWS).item() == 0.0

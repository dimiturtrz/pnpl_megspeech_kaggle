from typing import cast

import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.objective.auxiliary.semantic import Semantic
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.targets.speech import Speech

WIDTH = 16
DIM = 8
GENERATOR = torch.Generator().manual_seed(0)


class Words:
    def __init__(self, n: int) -> None:
        self.y = torch.zeros(n, dtype=torch.long)


def term() -> SpeechTerm:
    return SpeechTerm(2.0, Speech(torch.tensor([0, -1, 1, 2]), torch.randn(3, DIM, generator=GENERATOR).half()))


def test_head() -> None:
    head = term().head(WIDTH)
    assert isinstance(head, Semantic)
    assert head(torch.randn(2, 3, WIDTH, generator=GENERATOR)).shape == (2, 3, DIM)


def test_labels() -> None:
    assert term().labels(cast("Split", Words(4))).tolist() == [0, -1, 1, 2]
    with pytest.raises(ValueError, match="3 words against 4"):
        term().labels(cast("Split", Words(3)))


def test_unweighted() -> None:
    """Positive over the words with a target, differentiable into the projection, 0 when none has one."""
    aligned = term()
    head = aligned.head(WIDTH)
    projected = head(torch.randn(2, 3, WIDTH, generator=GENERATOR))
    value = aligned.unweighted(head, projected, torch.tensor([[0, 1, -100], [2, -1, -100]]))
    assert value.ndim == 0
    assert value.item() > 0
    value.backward()
    assert cast("Semantic", head).project.weight.grad is not None
    assert aligned.unweighted(head, projected, torch.full((2, 3), -1)).item() == 0.0

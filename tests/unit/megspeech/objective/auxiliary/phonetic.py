import math

import pytest
import torch

from megspeech.data.labels.phonemes import PHONEMES, UNKNOWN, Pronunciation
from megspeech.objective.auxiliary.phonetic import Phonetic, PhoneticTerm

WIDTH = 6
GENERATOR = torch.Generator().manual_seed(0)
TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


def test_forward() -> None:
    assert Phonetic(WIDTH).forward(torch.randn(2, 3, WIDTH, generator=GENERATOR)).shape == (2, 3, len(PHONEMES))
    assert Phonetic(WIDTH, 4).forward(torch.randn(2, WIDTH, generator=GENERATOR)).shape == (2, 4 * len(PHONEMES))


def term(positions: int = 1) -> PhoneticTerm:
    return PhoneticTerm(0.5, Pronunciation(torch.tensor([10, 20]), torch.tensor([[3, 1], [7, -1]])), positions)


def test_head() -> None:
    assert term().head(16)(torch.zeros(2, 16)).shape == (2, len(PHONEMES))
    assert term(2).head(16)(torch.zeros(2, 16)).shape == (2, 2 * len(PHONEMES))


def test_count() -> None:
    assert term().count() == len(PHONEMES)


def test_classify() -> None:
    """The leading `positions` phonemes of each token, unknown past a word's end or for a word not in the table."""
    assert torch.equal(term().classify(torch.tensor([[10, 20, 99]])), torch.tensor([[[3], [7], [UNKNOWN]]]))
    assert term(2).classify(torch.tensor([[20]]))[0, 0, 1].item() == UNKNOWN


def test_loss() -> None:
    """Uniform logits cost log(39) per known row, weighted; no known row costs nothing."""
    pronounced = term()
    logits = torch.zeros(2, 3, len(PHONEMES))
    head = pronounced.head(16)
    assert pronounced.loss(head, logits, TOKENS).item() == pytest.approx(0.5 * math.log(len(PHONEMES)))
    assert pronounced.loss(head, logits, torch.full((2, 3), 99)).item() == 0.0


def test_unweighted() -> None:
    """Two positions: the second is known for token 10 only, so 3 known rows of 6 average to log(39)."""
    pronounced = term(2)
    head = pronounced.head(16)
    logits = torch.zeros(2, 3, 2 * len(PHONEMES))
    assert pronounced.unweighted(head, logits, TOKENS).item() == pytest.approx(math.log(len(PHONEMES)))
    sure = torch.full((1, 1, 2 * len(PHONEMES)), -50.0)
    sure[0, 0, 3] = 50.0
    sure[0, 0, len(PHONEMES) + 1] = 50.0
    assert pronounced.unweighted(head, sure, torch.tensor([[10]])).item() == pytest.approx(0.0, abs=1e-6)

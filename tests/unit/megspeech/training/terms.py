import math
from typing import cast

import pytest
import torch

from megspeech.data.phonemes import PHONEMES, Pronunciation
from megspeech.objective.heads import DIM
from megspeech.objective.prototypes import Lexicon
from megspeech.objective.semantic import Semantic
from megspeech.training.terms import Aligned, Lexical, Pronounced

GENERATOR = torch.Generator().manual_seed(0)
TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


def aligned() -> Aligned:
    return Aligned(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def pronounced() -> Pronounced:
    return Pronounced(0.5, Pronunciation(torch.tensor([10, 20]), torch.tensor([[3, 1], [7, -1]])))


def test_aligned_head() -> None:
    assert isinstance(aligned().head(16), Semantic)


def test_pronounced_head() -> None:
    assert pronounced().head(16)(torch.zeros(2, 16)).shape == (2, len(PHONEMES))
    spelled = Pronounced(0.5, pronounced().pronunciation, 2)
    assert spelled.head(16)(torch.zeros(2, 16)).shape == (2, 2 * len(PHONEMES))


def test_aligned_loss() -> None:
    rule = aligned()
    head = rule.head(16)
    projected = head(torch.randn(2, 3, 16, generator=GENERATOR))
    value = rule.loss(head, projected, TOKENS)
    assert value.ndim == 0
    assert value.item() > 0
    value.backward()
    assert cast("Semantic", head).project.weight.grad is not None
    assert rule.loss(head, projected, torch.full((2, 3), 99)).item() == 0.0


def lexical() -> Lexical:
    return Lexical(3.0, aligned().lexicon)


def test_lexical_head() -> None:
    assert lexical().head(16)(torch.zeros(2, 16)).shape == (2, 2)


def test_lexical_loss() -> None:
    """Tokens 10 and 20 are the lexicon's 2 words; uniform logits cost log 2 per known row, unknown rows nothing."""
    rule = lexical()
    head = rule.head(16)
    assert rule.loss(head, torch.zeros(2, 3, 2), TOKENS).item() == pytest.approx(3.0 * math.log(2))
    assert rule.loss(head, torch.zeros(2, 3, 2), torch.full((2, 3), 99)).item() == 0.0
    sure = torch.tensor([[[50.0, -50.0], [-50.0, 50.0], [0.0, 0.0]]])
    assert rule.loss(head, sure, TOKENS[:1]).item() == pytest.approx(0.0, abs=1e-6)


def test_pronounced_loss() -> None:
    rule = pronounced()
    logits = torch.zeros(2, 3, len(PHONEMES))
    head = rule.head(16)
    assert rule.loss(head, logits, TOKENS).item() == pytest.approx(0.5 * math.log(len(PHONEMES)))
    assert rule.loss(head, logits, torch.full((2, 3), 99)).item() == 0.0


def test_pronounced_loss_positions() -> None:
    """Two positions: the second is known for token 10 only, so 3 known rows of 6 average to log(39)."""
    rule = Pronounced(1.0, pronounced().pronunciation, 2)
    logits = torch.zeros(2, 3, 2 * len(PHONEMES))
    assert rule.loss(rule.head(16), logits, TOKENS).item() == pytest.approx(math.log(len(PHONEMES)))
    sure = torch.full((1, 1, 2 * len(PHONEMES)), -50.0)
    sure[0, 0, 3] = 50.0
    sure[0, 0, len(PHONEMES) + 1] = 50.0
    assert rule.loss(rule.head(16), sure, torch.tensor([[10]])).item() == pytest.approx(0.0, abs=1e-6)

import math
from typing import cast

import pytest
import torch

from megspeech.data.phonemes import PHONEMES, Pronunciation
from megspeech.objective.heads import DIM
from megspeech.objective.prototypes import Lexicon
from megspeech.objective.semantic import Semantic
from megspeech.training.terms import Aligned, Pronounced

GENERATOR = torch.Generator().manual_seed(0)
TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


def aligned() -> Aligned:
    return Aligned(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def pronounced() -> Pronounced:
    return Pronounced(0.5, Pronunciation(torch.tensor([10, 20]), torch.tensor([3, 7])))


def test_aligned_head() -> None:
    assert isinstance(aligned().head(16), Semantic)


def test_pronounced_head() -> None:
    assert pronounced().head(16)(torch.zeros(2, 16)).shape == (2, len(PHONEMES))


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


def test_pronounced_loss() -> None:
    rule = pronounced()
    logits = torch.zeros(2, 3, len(PHONEMES))
    head = rule.head(16)
    assert rule.loss(head, logits, TOKENS).item() == pytest.approx(0.5 * math.log(len(PHONEMES)))
    assert rule.loss(head, logits, torch.full((2, 3), 99)).item() == 0.0

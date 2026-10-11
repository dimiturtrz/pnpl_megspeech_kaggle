import math
from typing import cast

import pytest
import torch

from megspeech.objective.auxiliary.semantic import Semantic, SemanticTerm
from megspeech.objective.scorer import BIAS, DIM, TEMPERATURE
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import Reading

WIDTH = 6
GENERATOR = torch.Generator().manual_seed(0)
TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


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
    loss = semantic.loss(targets, targets, torch.tensor([4, 7, 4]), torch.ones(3))
    assert loss.item() == pytest.approx(pair, rel=1e-5)
    padded = semantic.loss(
        torch.cat([targets, eye[2:]]), torch.cat([targets, eye[2:]]), torch.tensor([4, 7, 4, -1]), torch.ones(4)
    )
    assert padded.item() == pytest.approx(pair, rel=1e-5)
    assert semantic.loss(targets, targets, torch.full((3,), -100), torch.ones(3)).item() == 0.0


def test_softmax_loss() -> None:
    """Orthogonal targets: each row picks its own kind among the batch's distinct kinds; duplicates count once."""
    semantic = Semantic(WIDTH)
    eye = torch.eye(DIM)[:3]
    targets = eye[[0, 1, 0]]
    pair = softplus(-math.exp(TEMPERATURE))
    loss = semantic.softmax_loss(targets, targets, torch.tensor([4, 7, 4]), torch.ones(3))
    assert loss.item() == pytest.approx(pair, abs=1e-6)
    padded = semantic.softmax_loss(
        torch.cat([targets, eye[2:]]), torch.cat([targets, eye[2:]]), torch.tensor([4, 7, 4, -1]), torch.ones(4)
    )
    assert padded.item() == pytest.approx(pair, abs=1e-6)
    assert semantic.softmax_loss(targets, targets, torch.full((3,), -100), torch.ones(3)).item() == 0.0


def term() -> SemanticTerm:
    return SemanticTerm(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def test_head() -> None:
    assert isinstance(term().head(16), Semantic)


def test_unweighted() -> None:
    """Positive on the lexicon's words, differentiable into the projection, 0 when no token is in the lexicon."""
    aligned = term()
    head = aligned.head(16)
    projected = head(torch.randn(2, 3, 16, generator=GENERATOR))
    value = aligned.unweighted(aligned.read(head, projected, TOKENS))
    assert value.ndim == 0
    assert value.item() > 0
    value.backward()
    assert cast("Semantic", head).project.weight.grad is not None
    assert aligned.unweighted(aligned.read(head, projected, torch.full((2, 3), 99))).item() == 0.0


def test_unweighted_rows() -> None:
    """Rows at weight 1 are the unweighted loss; other weights move it; no token in the lexicon is still 0."""
    aligned = term()
    head = aligned.head(16)
    projected = head(torch.randn(2, 3, 16, generator=GENERATOR))
    assert torch.equal(
        aligned.unweighted(Reading(head, projected, TOKENS, torch.ones(2, 3))),
        aligned.unweighted(aligned.read(head, projected, TOKENS)),
    )
    skewed = aligned.unweighted(Reading(head, projected, TOKENS, torch.tensor([[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]])))
    assert skewed.item() != aligned.unweighted(aligned.read(head, projected, TOKENS)).item()
    assert aligned.unweighted(Reading(head, projected, torch.full((2, 3), 99), torch.full((2, 3), 3.0))).item() == 0.0

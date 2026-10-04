from itertools import combinations

import torch

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.topk import SmoothTopK

GENERATOR = torch.Generator().manual_seed(0)
LOSS = SmoothTopK(k=2, margin=1.0, tau=0.5)


def enumerated(logits: torch.Tensor, label: int) -> torch.Tensor:
    """Berrada's eq. 9 by listing every k-subset: the definition the recursion must reproduce."""
    k, tau = LOSS.k, LOSS.tau
    subsets = list(combinations(range(logits.shape[0]), k))
    scores = torch.stack([logits[list(subset)].sum() / (k * tau) for subset in subsets])
    missing = torch.tensor([LOSS.margin / tau * (label not in subset) for subset in subsets])
    holding = torch.tensor([label in subset for subset in subsets])
    return tau * (torch.logsumexp(scores + missing, 0) - torch.logsumexp(scores[holding], 0))


def test_call() -> None:
    logits = torch.randn(3, 5, generator=GENERATOR) * 3
    y = torch.tensor([0, 4, IGNORE])
    expected = torch.stack([enumerated(logits[0], 0), enumerated(logits[1], 4)]).mean()
    assert torch.allclose(LOSS(logits, y), expected, atol=1e-5)
    weight = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0])
    weighted = (3 * enumerated(logits[0], 0) + enumerated(logits[1], 4)) / 4
    assert torch.allclose(LOSS(logits, y, weight), weighted, atol=1e-5)


def test_call_gradient() -> None:
    logits = torch.randn(4, 50, generator=GENERATOR).requires_grad_()
    SmoothTopK()(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_call_satisfied() -> None:
    logits = torch.zeros(1, 5)
    logits[0, 0] = 100.0
    assert SmoothTopK(k=2, margin=1.0, tau=0.01)(logits, torch.tensor([0])) < 1e-3


def test_symmetric() -> None:
    logs = torch.randn(2, 5, generator=GENERATOR)
    table = LOSS.symmetric(logs)
    assert table.shape == (2, LOSS.k + 1)
    values = logs.exp()
    pairs = torch.stack([values[:, i] * values[:, j] for i, j in combinations(range(5), 2)]).sum(0)
    assert torch.allclose(table[:, 0], torch.zeros(2))
    assert torch.allclose(table[:, 1].exp(), values.sum(-1), atol=1e-5)
    assert torch.allclose(table[:, 2].exp(), pairs, atol=1e-5)

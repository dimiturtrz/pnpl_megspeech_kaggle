from itertools import combinations

import pytest
import torch
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.topk import SmoothTopK, TopKMix

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


def test_loss() -> None:
    """k times the paper's loss at temperature t/k and margin m/k: at k=1, t=1, m=0 it is the cross-entropy."""
    logits = torch.randn(4, 50, generator=GENERATOR) * 3
    y = torch.tensor([0, 7, 49, IGNORE])
    single = TopKMix(share=1.0, temperature=1.0, margin=0.0, k=1)
    assert single.loss == SmoothTopK(k=1, margin=0.0, tau=1.0)
    expected = functional.cross_entropy(logits, y, ignore_index=IGNORE)
    assert torch.allclose(single.k * single.loss(logits, y), expected, atol=1e-5)
    assert TopKMix(temperature=2.0, margin=3.0, k=10).loss == SmoothTopK(k=10, margin=0.3, tau=0.2)


def test_mixed_off() -> None:
    """Share 0 returns the head's own loss object itself, whatever the top-k term would be."""
    own = torch.tensor(2.5)
    assert TopKMix().mixed(own, torch.full((2, 50), torch.nan), torch.tensor([0, 1])) is own


def test_mixed() -> None:
    """A word ranked first by a margin over the 10th competitor costs ~0 at share 1; a share blends linearly."""
    logits = torch.zeros(2, 50)
    logits[0, 3] = logits[1, 9] = 40.0
    y = torch.tensor([3, 9])
    own = torch.tensor(1.0)
    assert TopKMix(share=1.0).mixed(own, logits, y) < 1e-6
    half = TopKMix(share=0.5)
    assert torch.allclose(half.mixed(own, logits, y), torch.tensor(0.5), atol=1e-6)
    missed = torch.zeros(1, 50)
    missed[0, :10] = 5.0
    assert TopKMix(share=1.0).mixed(own, missed, torch.tensor([20])) > 1.0


@pytest.mark.parametrize("scale", [1.0, 30.0, 300.0])
def test_mixed_gradient(scale: float) -> None:
    """Finite gradients at the logit ranges a bfloat16 head emits, its rounded values included."""
    logits = (torch.randn(8, 50, generator=GENERATOR) * scale).bfloat16().float().requires_grad_()
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, IGNORE])
    TopKMix(share=0.5).mixed(functional.cross_entropy(logits, y, ignore_index=IGNORE), logits, y).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

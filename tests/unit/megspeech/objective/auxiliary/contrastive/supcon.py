from typing import override

import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.auxiliary.contrastive.similarity import Similarity
from megspeech.objective.auxiliary.contrastive.supcon import Positives, SupConTerm
from megspeech.objective.term import Reading

TARGET = torch.tensor([[0, 1, 2, IGNORE]])


class Everyone(Positives):
    """Every labelled pair but the slot itself is a candidate; slots 0 and 1 are each other's positive."""

    @override
    def pairs(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target != IGNORE
        candidate = valid[:, None] & valid[None, :] & ~torch.eye(len(target), dtype=torch.bool)
        positive = torch.zeros_like(candidate)
        positive[0, 1] = positive[1, 0] = True
        return candidate, positive & candidate


class Plain(SupConTerm):
    @override
    def positives(self) -> Positives:
        return Everyone()

    @override
    def labels(self, split: Split) -> torch.Tensor:
        return split.y


def test_pairs() -> None:
    """`pairs` is the strategy's."""
    candidate, positive = Plain(weight=1.0).pairs(TARGET.flatten())
    assert candidate.sum() == 6
    assert positive.nonzero().tolist() == [[0, 1], [1, 0]]


def test_positives() -> None:
    assert isinstance(Plain(weight=1.0).positives(), Everyone)


def test_head() -> None:
    assert isinstance(Plain(weight=1.0).head(4), Similarity)


def test_unweighted() -> None:
    """Closer positives cost less; a batch with no positive costs exactly 0 with finite gradients."""
    term, rows = Plain(weight=1.0), torch.ones(TARGET.shape)
    apart = torch.eye(4)[None, :3] * 2
    apart = torch.cat([apart, torch.zeros(1, 1, 4)], dim=1)
    together = apart.clone()
    together[0, 1] = together[0, 0]
    assert term.unweighted(Reading(Similarity(), together, TARGET, rows)) < term.unweighted(
        Reading(Similarity(), apart, TARGET, rows)
    )
    lonely = torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(0)).requires_grad_()
    cost = term.unweighted(Reading(Similarity(), lonely, torch.tensor([[0, IGNORE, 2]]), torch.ones(1, 3)))
    cost.backward()
    assert cost.item() == 0.0
    assert lonely.grad is not None
    assert torch.isfinite(lonely.grad).all()

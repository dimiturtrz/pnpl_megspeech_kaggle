import torch

from megspeech.evaluation.metric.balance import Sinkhorn

GENERATOR = torch.Generator().manual_seed(0)


def column_mass(scores: torch.Tensor) -> torch.Tensor:
    return scores.log_softmax(-1).logsumexp(0)


def test_sinkhorn_zero_steps_is_none() -> None:
    scores = torch.randn(4, 3, generator=GENERATOR)
    assert torch.equal(Sinkhorn()(scores), scores)


def test_sinkhorn_equalises_class_mass() -> None:
    """A set biased toward one class ends with every class holding the same posterior mass."""
    scores = torch.randn(200, 5, generator=GENERATOR) + torch.tensor([3.0, 0.0, 0.0, 0.0, -2.0])
    mass = column_mass(Sinkhorn(50)(scores))
    assert torch.allclose(mass, mass.mean().expand_as(mass), atol=1e-3)


def test_sinkhorn_keeps_balanced_set() -> None:
    """A set whose classes already share the mass equally is only shifted by a constant."""
    scores = torch.eye(4) * 2.0
    assert torch.allclose(Sinkhorn(50)(scores), scores, atol=1e-5)


def test_sinkhorn_one_step_moves_toward_balance() -> None:
    scores = torch.randn(200, 5, generator=GENERATOR) + torch.tensor([3.0, 0.0, 0.0, 0.0, -2.0])
    assert column_mass(Sinkhorn(1)(scores)).std() < column_mass(scores).std()

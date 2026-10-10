import torch

from megspeech.evaluation.metric.balance import Sinkhorn

PRIOR = torch.log(torch.tensor([0.5, 0.25, 0.15, 0.1]))
SCORES = torch.arange(24, dtype=torch.float32).reshape(6, 4).sin() * 2.0


def column_mass(scores: torch.Tensor) -> torch.Tensor:
    return scores.log_softmax(-1).logsumexp(0)


def reference(scores: torch.Tensor, steps: int, toward: float) -> torch.Tensor:
    """The probability-space algorithm: columns to `n * softmax(toward * log_prior)`, then rows to 1."""
    p = scores.softmax(-1)
    ratio = (toward * PRIOR).softmax(-1)
    for _ in range(steps):
        p = p / p.sum(0) * (len(p) * ratio)
        p = p / p.sum(1, keepdim=True)
    return p


def test_sinkhorn_zero_steps_is_none() -> None:
    assert torch.equal(Sinkhorn()(SCORES, PRIOR), SCORES)


def test_sinkhorn_equalises_class_mass() -> None:
    mass = column_mass(Sinkhorn(50)(SCORES, PRIOR))
    assert torch.allclose(mass, mass.mean().expand_as(mass), atol=1e-4)


def test_sinkhorn_keeps_balanced_set() -> None:
    scores = torch.eye(4) * 2.0
    assert torch.allclose(Sinkhorn(50)(scores, PRIOR), scores, atol=1e-5)


def test_sinkhorn_one_step_moves_toward_balance() -> None:
    assert column_mass(Sinkhorn(1)(SCORES, PRIOR)).std() < column_mass(SCORES).std()


def test_sinkhorn_matches_probability_space() -> None:
    """The logit shift and the textbook probability-space iteration give the same posteriors."""
    for toward in (0.0, 0.25, 1.0):
        shifted = Sinkhorn(30, toward)(SCORES, PRIOR).softmax(-1)
        assert torch.allclose(shifted, reference(SCORES, 30, toward), atol=1e-5)


def test_sinkhorn_toward_sets_class_mass_share() -> None:
    """At `toward` the converged class masses follow `softmax(toward * log_prior)`."""
    mass = column_mass(Sinkhorn(200, 0.5)(SCORES, PRIOR)).exp()
    assert torch.allclose(mass / mass.sum(), (0.5 * PRIOR).softmax(-1), atol=1e-4)

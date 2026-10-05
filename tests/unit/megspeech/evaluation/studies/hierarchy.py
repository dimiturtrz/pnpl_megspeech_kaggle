import numpy as np
import pytest
import torch

from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Score
from megspeech.evaluation.studies.hierarchy import GRID, THRESHOLDS, Domain, Gate, Grid, Hierarchical, WordTags

LOG_PRIOR = torch.zeros(VOCAB_SIZE)


def domain(seed: int, n: int = 100) -> Domain:
    """Noisy logits; a bonus that is high on the true word."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % VOCAB_SIZE
    logits = torch.randn(n, VOCAB_SIZE, generator=generator)
    bonus = torch.zeros(n, VOCAB_SIZE)
    bonus[torch.arange(n), labels] = 5.0
    return Domain(logits, bonus, labels)


def score(by_tau: float, n: int = 10) -> Score:
    return Score.best({0.0: by_tau, 0.5: 0.0, 1.0: 0.0}, n)


def test_word_tags_of() -> None:
    tags = WordTags.of(np.array([0, 0, 0, 1, -100]), np.array([1, 1, UNKNOWN, 0, 0]), 2)
    assert tags.distribution.shape == (VOCAB_SIZE, 2)
    assert tags.distribution[0].tolist() == pytest.approx([1 / 4, 3 / 4])
    assert tags.distribution[1].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert tags.distribution[2].tolist() == pytest.approx([0.5, 0.5])
    assert tags.unknown[:3].tolist() == pytest.approx([0.4, 1 / 3, 0.5])


def test_bonus() -> None:
    tags = WordTags.of(np.array([0, 0, 0, 1, -100]), np.array([1, 1, UNKNOWN, 0, 0]), 2)
    bonus = tags.bonus(np.log(np.array([[0.9, 0.1]])), np.array([0.5, 0.5]))
    assert bonus.shape == (1, VOCAB_SIZE)
    assert bonus[0, 0] == pytest.approx(np.log(0.4 + 0.6 * 0.6), abs=1e-4)
    assert bonus[0, 2] == pytest.approx(0.0, abs=1e-4)


def test_hierarchical_fuse() -> None:
    logits, bonus = torch.zeros(1, 3), torch.tensor([[0.0, -1.0, 1.0]])
    assert torch.allclose(Hierarchical().fuse(logits, bonus, 2.0), 2 * bonus)
    assert torch.equal(Hierarchical().fuse(logits, bonus, 0.0), logits)


def test_gate_fuse() -> None:
    logits, bonus = torch.zeros(1, 3), torch.tensor([[0.0, -1.0, 1.0]])
    assert Gate().fuse(logits, bonus, 1.0).tolist() == [[0.0, -1000.0, 0.0]]
    assert torch.equal(Gate().fuse(logits, bonus, 0.0), logits)


def test_fused() -> None:
    held = domain(0, 4)
    assert torch.equal(held.fused(Hierarchical(), 0.0), held.logits)
    assert torch.allclose(held.fused(Hierarchical(), 2.0), held.logits + 2 * held.bonus)
    assert torch.equal(held.fused(Gate(), 0.0), held.logits)


def test_grid_of() -> None:
    evaluator, domains = Evaluator(torch.device("cpu")), {"a": domain(0), "b": domain(1)}
    grid = Grid.of(evaluator, LOG_PRIOR, domains, Hierarchical())
    assert set(grid.scores) == {"a", "b"}
    assert set(grid.scores["a"]) == set(GRID)
    assert grid.scores["a"][4.0].bacc10 > grid.scores["a"][0.0].bacc10
    assert set(Grid.of(evaluator, LOG_PRIOR, domains, Gate()).scores["a"]) == set(THRESHOLDS)


def test_settings() -> None:
    assert Grid({"a": {lam: score(0.2) for lam in GRID}}).settings == GRID


def test_macro() -> None:
    grid = Grid({"a": {lam: score(0.2) for lam in GRID}, "b": {lam: score(0.6) for lam in GRID}})
    macro = grid.macro(0.0)
    assert macro.bacc10 == pytest.approx(0.4)
    assert macro.n == 20


def test_chosen() -> None:
    better = {lam: score(lam / 10) for lam in GRID}
    flat = {lam: score(0.3) for lam in GRID}
    grid = Grid({"a": better, "b": better, "c": flat})
    assert grid.chosen("c") == max(GRID)
    assert Grid({"a": flat, "b": flat}).chosen("a") == 0.0


def test_leave_one_out() -> None:
    better = {lam: score(lam / 10) for lam in GRID}
    macro, weights = Grid({"a": better, "b": better, "c": better}).leave_one_out()
    assert weights == {"a": 4.0, "b": 4.0, "c": 4.0}
    assert macro.bacc10 == pytest.approx(0.4)


def test_per_domain() -> None:
    better = {lam: score(lam / 10) for lam in GRID}
    grid = Grid({"a": better, "b": better})
    per = grid.per_domain({"a": 4.0, "b": 0.0}, grid.macro(0.0))
    assert per == {"a": (0.4, 10), "b": (0.0, 10)}

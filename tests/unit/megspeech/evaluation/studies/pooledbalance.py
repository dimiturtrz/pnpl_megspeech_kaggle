import pytest
import torch

from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.studies.pooledbalance import GROUPS, PooledBalance, Unseen

CLASSES = 40
ROWS = 400


def study() -> PooledBalance:
    evaluator = Evaluator(device=torch.device("cpu"), balance=Sinkhorn(steps=30))
    return PooledBalance(evaluator, torch.zeros(CLASSES))


def parts() -> dict[int, Unseen]:
    generator = torch.Generator().manual_seed(0)
    y = torch.arange(ROWS) % CLASSES
    shift = torch.zeros(CLASSES)
    shift[:15] = 3.0
    return {
        1: Unseen(torch.randn(ROWS, CLASSES, generator=generator) + 1.5 * torch.eye(CLASSES)[y], y),
        2: Unseen(torch.randn(ROWS, CLASSES, generator=generator) + 1.5 * torch.eye(CLASSES)[y] + shift, y),
    }


def test_alone() -> None:
    result = study().alone(parts())
    assert set(result) == {0.0, 0.5, 1.0}
    assert set(result[0.0]) == {1, 2}


def test_pooled_differs_when_subjects_are_shifted() -> None:
    alone, pooled = study().alone(parts()), study().pooled(parts())
    assert pooled[0.0] != alone[0.0]


def test_pooled_equals_alone_for_one_subject() -> None:
    single = {1: parts()[1]}
    assert study().pooled(single) == study().alone(single)


def test_bacc() -> None:
    assert study().bacc(torch.eye(CLASSES), torch.arange(CLASSES)) == 1.0


def test_macro() -> None:
    macro = PooledBalance.macro({0.0: {1: 0.2, 2: 0.4}, 0.5: {1: 0.5, 2: 0.7}, 1.0: {1: 0.1, 2: 0.1}})
    assert macro.tau == 0.5
    assert macro.bacc10 == pytest.approx(0.6)


def test_grouped() -> None:
    grouped = PooledBalance.grouped({0.5: {1: 0.2, 2: 0.4, 13: 0.5, 25: 0.9}}, 0.5)
    assert grouped["1-12"] == pytest.approx(0.3)
    assert grouped["13-22"] == 0.5
    assert grouped["23-32"] == 0.9
    assert set(grouped) == set(GROUPS)


def test_report() -> None:
    text = study().report(parts())
    assert "per-subject" in text
    assert "pooled" in text

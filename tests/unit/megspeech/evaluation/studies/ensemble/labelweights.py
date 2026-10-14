from megspeech.evaluation.labelweighted import Shrinkage
from megspeech.evaluation.studies.ensemble.labelweights import CrossFit
from tests.unit.megspeech.evaluation.labelweighted import pool


def test_halves() -> None:
    assert CrossFit.halves([1, 2, 3, 4, 5]) == ([1, 3, 5], [2, 4])


def test_grid() -> None:
    assert len(CrossFit.grid()) == len({(s.k, s.beta) for s in CrossFit.grid()})


def test_held_out() -> None:
    p = pool()
    assert 0.0 <= CrossFit(p).held_out([1, 2, 3], [4, 5, 6], Shrinkage(5.0, 10.0)) <= 1.0


def test_inner() -> None:
    p = pool()
    assert 0.0 <= CrossFit(p).inner([1, 2, 3, 4], Shrinkage(5.0, 10.0)) <= 1.0


def test_choose() -> None:
    p = pool()
    assert CrossFit(p).choose([1, 2, 3, 4]) in CrossFit.grid()


def test_fold() -> None:
    """The weighted score is the chosen grid point's held-out score."""
    fold = CrossFit(pool()).fold([1, 3, 5], [2, 4, 6])
    assert fold.weighted == fold.every[fold.chosen]


def test_run() -> None:
    assert len(CrossFit(pool()).run()) == 2

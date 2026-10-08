import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.studies.labelweights import CrossFit, Pool, Shrinkage

MEMBERS = 3
WORDS = 400
SUBJECTS = 6


def pool() -> Pool:
    generator = torch.Generator().manual_seed(0)
    labels = {s: torch.randint(VOCAB_SIZE, (WORDS,), generator=generator) for s in range(1, SUBJECTS + 1)}
    logp = {s: torch.randn(MEMBERS, WORDS, VOCAB_SIZE, generator=generator).log_softmax(-1) for s in labels}
    return Pool(labels, logp, torch.zeros(VOCAB_SIZE), Evaluator(device=torch.device("cpu"), min_novel=1))


def test_weights() -> None:
    """Weights sum to one over members for every label; beta 0 is uniform and a better member gains weight."""
    p = pool()
    subjects = p.subjects()
    assert torch.allclose(Shrinkage(5.0, 30.0).weights(p, subjects).sum(0), torch.ones(VOCAB_SIZE))
    assert torch.allclose(Shrinkage(5.0, 0.0).weights(p, subjects), p.uniform())
    for s in subjects:
        p.logp[s][0] = p.logp[s][0].scatter(1, p.labels[s][:, None], 0.0)
    assert (Shrinkage(5.0, 30.0).weights(p, subjects).argmax(0) == 0).all()


def test_blended() -> None:
    """Uniform weights give the members' mean log-probability."""
    p = pool()
    assert torch.allclose(p.blended(1, p.uniform()), p.logp[1].mean(0), atol=1e-6)


def test_uniform() -> None:
    assert tuple(pool().uniform().shape) == (MEMBERS, VOCAB_SIZE)


def test_subjects() -> None:
    assert pool().subjects() == list(range(1, SUBJECTS + 1))


def test_macro() -> None:
    """A macro is a balanced accuracy in [0, 1]."""
    p = pool()
    assert 0.0 <= p.macro(p.subjects(), p.uniform()).bacc10 <= 1.0


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

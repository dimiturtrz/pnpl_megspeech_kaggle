import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.labelweighted import Pool, Shrinkage
from megspeech.evaluation.scoring.evaluator import Evaluator

MEMBERS = 3
WORDS = 400
SUBJECTS = 6
TEST_ROWS = 20


def pool(*, kaggle: bool = False) -> Pool:
    generator = torch.Generator().manual_seed(0)
    labels = {s: torch.randint(VOCAB_SIZE, (WORDS,), generator=generator) for s in range(1, SUBJECTS + 1)}
    logp = {s: torch.randn(MEMBERS, WORDS, VOCAB_SIZE, generator=generator).log_softmax(-1) for s in labels}
    tests = torch.randn(MEMBERS, TEST_ROWS, VOCAB_SIZE, generator=generator).log_softmax(-1) if kaggle else None
    return Pool(labels, logp, torch.zeros(VOCAB_SIZE), Evaluator(device=torch.device("cpu"), min_novel=1), tests)


def test_weights() -> None:
    """Weights sum to one over members for every label; beta 0 is uniform and a better member gains weight."""
    p = pool()
    subjects = p.subjects()
    assert torch.allclose(Shrinkage(5.0, 30.0).weights(p, subjects).sum(0), torch.ones(VOCAB_SIZE))
    assert torch.allclose(Shrinkage(5.0, 0.0).weights(p, subjects), p.uniform())
    for s in subjects:
        p.logp[s][0] = p.logp[s][0].scatter(1, p.labels[s][:, None], 0.0)
    assert (Shrinkage(5.0, 30.0).weights(p, subjects).argmax(0) == 0).all()


def test_entropy() -> None:
    """Uniform weights are 1, one member alone 0."""
    uniform = pool().uniform()
    alone = torch.zeros(MEMBERS, VOCAB_SIZE)
    alone[0] = 1.0
    assert torch.allclose(Shrinkage.entropy(uniform), torch.ones(VOCAB_SIZE))
    assert torch.allclose(Shrinkage.entropy(alone), torch.zeros(VOCAB_SIZE), atol=1e-6)


def test_blended() -> None:
    """Uniform weights give the members' mean log-probability."""
    p = pool()
    assert torch.allclose(p.blended(1, p.uniform()), p.logp[1].mean(0), atol=1e-6)


def test_tested() -> None:
    """Uniform weights give the members' mean Kaggle log-probability; a pool read without a track refuses."""
    p = pool(kaggle=True)
    assert p.kaggle is not None
    assert torch.allclose(p.tested(p.uniform()), p.kaggle.mean(0), atol=1e-6)
    with pytest.raises(ValueError, match="without a track"):
        pool().tested(p.uniform())


def test_finished() -> None:
    """Each Kaggle test row is a distribution."""
    p = pool(kaggle=True)
    assert torch.allclose(p.finished(p.uniform(), 1.0).sum(-1), torch.ones(TEST_ROWS))


def test_uniform() -> None:
    assert tuple(pool().uniform().shape) == (MEMBERS, VOCAB_SIZE)


def test_subjects() -> None:
    assert pool().subjects() == list(range(1, SUBJECTS + 1))


def test_macro() -> None:
    """A macro is a balanced accuracy in [0, 1]."""
    p = pool()
    assert 0.0 <= p.macro(p.subjects(), p.uniform()).bacc10 <= 1.0

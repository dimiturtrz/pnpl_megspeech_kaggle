import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.spans.spans import Spans
from megspeech.training.sampling.curriculum import Annealed, Balanced, Base, Natural, Prioritised, Robust, Uniform
from megspeech.training.sampling.sampler import Sampler, SamplerConfig

TRAIN = Split.of(
    Windows.stacked(torch.zeros(6, 1, 2)),
    (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)),
    Spans(np.array([[0, 1, 2], [3, -1, -1], [4, 5, -1]]), np.array([[0, 0.2, 0.4], [0, 0, 0], [0, 1.0, 0]])),
    np.array([1.0, 3.0, 2.0, 5.0, 1.0, 2.0]),
    np.array([0, 0, 0, 1, 1, 1]),
)
SEED = 7
DRAWS = 64


def draw(weights: torch.Tensor) -> list[int]:
    torch.manual_seed(SEED)
    return torch.multinomial(weights, DRAWS, replacement=True).tolist()


def test_of() -> None:
    """The default config's chain is today's `Natural`: bit-identical weights, so bit-identical draws."""
    made = Sampler.of(SamplerConfig(), TRAIN)
    assert isinstance(made, Natural)
    old = TRAIN.weight[TRAIN.index[:, 0]]
    assert torch.equal(made.weights(TRAIN, 3, 10), old)
    assert draw(made.weights(TRAIN, 3, 10)) == draw(old)


def test_scheduled_matches_annealed_natural() -> None:
    """Annealed over the default base is today's `Annealed`: the pre-refactor formula, recomputed here."""
    made = Sampler.scheduled(SamplerConfig(curriculum=0.9), TRAIN)
    assert isinstance(made, Annealed)
    first = TRAIN.index[:, 0]
    weight, subject = TRAIN.weight[first], TRAIN.subject[first]
    present, which = torch.unique(subject, return_inverse=True)
    totals = torch.zeros(len(present), dtype=weight.dtype).index_add_(0, which, weight)
    progress = 4 / 9
    deep = (1 - progress) / len(present) + progress * 0.9
    shares = torch.where(present == 0, deep, (1 - deep) / (len(present) - 1))
    old = weight / totals[which] * shares[which]
    assert torch.equal(made.weights(TRAIN, 4, 10), old)
    assert draw(made.weights(TRAIN, 4, 10)) == draw(old)


def test_of_robust_and_prioritised_match_the_old_chain() -> None:
    """Robust then Prioritised over Natural: the new chain's first draw weights equal the old stack's, and a
    prioritised-only config reproduces the old floor/ceiling formula on the same observed losses."""
    optim = SamplerConfig(robust=1.0, priority=1.0, visibility=1.0)
    made = Sampler.of(optim, TRAIN)
    assert isinstance(made, Prioritised)
    assert isinstance(made.base, Robust)
    assert isinstance(made.base.base, Natural)
    base = TRAIN.weight[TRAIN.index[:, 0]]
    assert torch.equal(made.weights(TRAIN, 0, 10), base)
    made.difficulty = torch.tensor([1.0, 3.0, 2.0])
    mean = made.difficulty.mean()
    expected = torch.minimum(torch.maximum(base * (made.difficulty / mean), 0.25 * base), 4.0 * base)
    assert torch.equal(made.weights(TRAIN, 1, 10), expected)
    assert draw(made.weights(TRAIN, 1, 10)) == draw(expected)


def test_of_base_options() -> None:
    assert Sampler.of(SamplerConfig(base=Base.UNIFORM), TRAIN) == Uniform()
    assert Sampler.of(SamplerConfig(base=Base.STRATIFIED), TRAIN).weights(TRAIN, 0, 10).shape == (3,)


@pytest.mark.parametrize("sampler", list(Base))
def test_scheduled(sampler: Base) -> None:
    made = Sampler.scheduled(SamplerConfig(base=sampler), TRAIN)
    assert (made.weights(TRAIN, 0, 10) > 0).all()


def test_scheduled_balanced() -> None:
    """A `balance` wraps the base in `Balanced` beneath the annealing; 0 leaves the base bare."""
    assert isinstance(Sampler.scheduled(SamplerConfig(balance=0.5), TRAIN), Balanced)
    annealed = Sampler.scheduled(SamplerConfig(balance=0.5, curriculum=0.9), TRAIN)
    assert isinstance(annealed, Annealed)
    assert isinstance(annealed.base, Balanced)
    assert isinstance(Sampler.scheduled(SamplerConfig(balance=0.0), TRAIN), Natural)

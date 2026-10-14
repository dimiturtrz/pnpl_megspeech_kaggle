import math

import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.refiner.belief import STATES
from megspeech.evaluation.studies.refiner.coupling import (
    EDGE,
    LAGS,
    RADIX,
    Cells,
    Context,
    Correlated,
    Coupling,
    Profile,
    Quality,
)
from megspeech.evaluation.studies.refiner.rows import Real
from megspeech.evaluation.studies.refiner.wiki import OOV

WIDTH = 6
SPANS = 34


def every_state() -> Real:
    """Every state twice over 17 full spans of 6, beliefs leaning on the true word."""
    states = torch.arange(SPANS * WIDTH) % STATES
    torch.manual_seed(0)
    beliefs = torch.randn(len(states), VOCAB_SIZE)
    words = states < OOV
    beliefs[words, states[words]] += 3.0
    return Real.of(beliefs, torch.where(words, states, -100), torch.arange(len(states)).view(-1, WIDTH))


def test_context_of() -> None:
    context = Context.of(torch.tensor([[3, 4, PAD]]))
    assert context.left.tolist() == [[EDGE, 3, 4]]
    assert context.own.tolist() == [[3, 4, EDGE]]
    assert context.right.tolist() == [[4, EDGE, EDGE]]


def test_context_cells() -> None:
    triple, pair, own = Context.of(torch.tensor([[3, 4]])).cells()
    assert own.tolist() == [[3, 4]]
    assert pair.tolist() == [[3 * RADIX + 4, 4 * RADIX + EDGE]]
    assert triple[0, 0] == EDGE * RADIX * RADIX + 3 * RADIX + 4


def test_cells_of() -> None:
    keys = torch.tensor([1, 0, 1, 0])
    quality = torch.tensor([0.5, 0.9, 0.1, 0.2])
    cells = Cells.of(keys, quality, torch.zeros(4, 2), keys)
    assert cells.order.tolist() == [3, 1, 2, 0]
    assert cells.counts.tolist() == [2, 2]
    assert cells.kappa == 0.0


def test_shrinkage() -> None:
    keys = torch.tensor([0, 0, 1, 1])
    assert Cells.shrinkage(keys, keys, torch.randn(4, 3)) == 0.0


def test_shrinkage_separated_cells_trusted() -> None:
    keys = torch.tensor([0, 0, 1, 1])
    beliefs = torch.tensor([[0.0], [0.1], [10.0], [10.1]])
    assert Cells.shrinkage(keys, torch.zeros(4, dtype=torch.long), beliefs) < 1e-2


def test_shrinkage_identical_cells_untrusted() -> None:
    keys = torch.tensor([0, 0, 1, 1])
    beliefs = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    assert Cells.shrinkage(keys, torch.zeros(4, dtype=torch.long), beliefs) == math.inf


def test_find() -> None:
    keys = torch.tensor([2, 2, 5])
    cells = Cells.of(keys, torch.zeros(3), torch.zeros(3, 1), keys)
    start, count = cells.find(torch.tensor([[5, 2, 7]]))
    assert count.tolist() == [[1, 2, 0]]
    assert start[0, :2].tolist() == [2, 0]


def test_fit() -> None:
    lags = torch.arange(1, LAGS + 1).double()
    truth = 0.1 + 0.6 * 0.4**lags
    fitted = Coupling.fit(truth.float(), torch.ones(LAGS))
    assert fitted.a == pytest.approx(0.1, abs=1e-2)
    assert fitted.b == pytest.approx(0.6, abs=2e-2)
    assert fitted.phi == pytest.approx(0.4, abs=1e-2)


def test_fit_skips_unmeasured() -> None:
    lags = torch.arange(1, LAGS + 1).double()
    truth = (0.1 + 0.6 * 0.4**lags).float()
    truth[-2:] = math.nan
    assert Coupling.fit(truth, torch.ones(LAGS)).phi == pytest.approx(0.4, abs=1e-2)


def test_fit_uncorrelated() -> None:
    fitted = Coupling.fit(torch.zeros(LAGS), torch.ones(LAGS))
    assert fitted.a + fitted.b == pytest.approx(0.0, abs=1e-6)


def test_quantiles() -> None:
    u = Coupling(0.1, 0.6, 0.4).quantiles((20000, 3), torch.Generator().manual_seed(0))
    z = torch.special.ndtri(u)
    lag1 = torch.corrcoef(torch.stack([z[:, 0], z[:, 1]]))[0, 1]
    assert float(lag1) == pytest.approx(0.1 + 0.6 * 0.4, abs=0.03)
    assert bool(((u > 0) & (u < 1)).all())


def test_quantiles_independent() -> None:
    z = torch.special.ndtri(Coupling(0.0, 0.0, 0.0).quantiles((20000, 2), torch.Generator().manual_seed(0)))
    assert abs(float(torch.corrcoef(z.T)[0, 1])) < 0.03


def test_raw() -> None:
    beliefs = torch.full((2, VOCAB_SIZE), -9.0)
    beliefs[0, 1], beliefs[1, 0] = -2.0, -0.5
    real = Real(beliefs, torch.tensor([1, OOV]), torch.tensor([[0, 1]]))
    assert Quality(real).raw().tolist() == [-2.0, 0.5]


def test_scores() -> None:
    beliefs = torch.tensor([[0.0, -9.0], [-1.0, -9.0], [-2.0, -9.0]])
    real = Real(beliefs, torch.tensor([0, 0, 0]), torch.tensor([[0, 1]]))
    scores = Quality(real).scores()
    assert scores[0] > 0 > scores[1]
    assert scores[2].isnan()


def test_lagged() -> None:
    correlation, pairs = Quality(every_state()).lagged()
    assert correlation.shape == (LAGS,)
    assert pairs[0] > pairs[1]


def test_correlated_of() -> None:
    sampler = Correlated.of(every_state())
    assert len(sampler.levels) == 3
    assert len(sampler.levels[-1].keys) == STATES


def test_correlated_of_missing_state() -> None:
    real = every_state()
    with pytest.raises(ValueError, match="no in-span row"):
        Correlated.of(real.take(torch.arange(SPANS) < 1))


def test_draw() -> None:
    real = every_state()
    sampler = Correlated.of(real)
    states = torch.tensor([[3, 4, OOV], [7, PAD, PAD]])
    drawn = sampler.draw(states, torch.Generator().manual_seed(0))
    assert drawn.shape == (2, 3, VOCAB_SIZE)
    assert bool((drawn[1, 1:] == 0).all())
    rows = (real.beliefs[:, None, :] == drawn[0, 0][None, None]).all(-1).squeeze(1).nonzero()
    assert real.states[rows].flatten().tolist() == [3] * len(rows)


def test_profile_of() -> None:
    profile = Profile.of(every_state())
    assert profile.own > profile.right


def test_given() -> None:
    real = Real(torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]), torch.tensor([0, 1]), torch.tensor([[0, 1]]))
    assert Profile.given(real, real.index[:, :1], real.index[:, 1:]) == -2.0


def test_simulated() -> None:
    real = every_state()
    simulated = Profile.simulated(real, Correlated.of(real), 0)
    assert simulated.own == pytest.approx(Profile.of(real).own, abs=1.0)


def test_contextual() -> None:
    sampler = Correlated.contextual(every_state())
    assert sampler.coupling == Coupling(0.0, 0.0, 0.0)
    assert len(sampler.levels) == 3


def test_coupled() -> None:
    sampler = Correlated.coupled(every_state())
    assert len(sampler.levels) == 1
    assert sampler.draw(torch.tensor([[3, OOV]]), torch.Generator().manual_seed(0)).shape == (1, 2, VOCAB_SIZE)

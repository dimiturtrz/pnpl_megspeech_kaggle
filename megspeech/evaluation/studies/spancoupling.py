"""A correlated span sampler for the refiner's simulated beliefs (7gf.145.1, "not ruled out" item 1).

`Bank` draws each word's belief independently given its own state. Real out-of-fold beliefs are not like that, in two
measured ways. The context head reads the whole span, so a row's belief depends on its neighbours' states (`Cells`):
each row is drawn from the real rows sharing its (left, own, right) states, backing off to (own, right) and then to own
alone with the empirical-Bayes weight `n / (n + kappa)`, `kappa` the ratio of within-cell to between-cell belief
variance measured on the bank (Robbins/Efron-Morris shrinkage: a cell is trusted as far as its mean is estimated
better than the parent's). And a span's rows err together (`Coupling`): the within-state rank of each row's quality
(log-probability of its true state, or minus its top log-probability for `OOV`) correlates along the span as
`a + b * phi**lag` -- a span-wide share `a` and a decaying local share `b` -- fitted to the lag correlations measured
between in-vocabulary rows (an `OOV` row's quality has no true word to read, so it is ranked but not measured); a
Gaussian copula with that law picks each row's quantile within its cell.

`Profile` is the check that the sampler reproduces what it was fitted to, on the held-out fold it never saw;
`lmrefine --sampler correlated` records it.
"""

import math
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.spans.spans import MAX_WORDS, PAD
from megspeech.evaluation.studies.drawer import Drawer
from megspeech.evaluation.studies.refiner import STATES
from megspeech.evaluation.studies.refinerows import Real
from megspeech.evaluation.studies.wikispans import OOV

EDGE = STATES
"""The context state beyond a span's ends."""
RADIX = STATES + 1
GRID = 1000
"""Resolution of the decay `phi` the coupling fit scans over (0, 1)."""
LAGS = MAX_WORDS - 1


@dataclass(frozen=True)
class Context:
    """Each position's (left, own, right) states in spans of states, `EDGE` past the ends."""

    left: Int[Tensor, "b w"]
    own: Int[Tensor, "b w"]
    right: Int[Tensor, "b w"]

    @classmethod
    def of(cls, states: Int[Tensor, "b w"]) -> "Context":
        own = torch.where(states == PAD, EDGE, states.long())
        edge = torch.full_like(own[:, :1], EDGE)
        return cls(torch.cat([edge, own[:, :-1]], 1), own, torch.cat([own[:, 1:], edge], 1))

    def cells(self) -> tuple[Int[Tensor, "b w"], ...]:
        """Each position's cell key at every level, finest first: (left, own, right), (own, right), own."""
        pair = self.own * RADIX + self.right
        return self.left * RADIX * RADIX + pair, pair, self.own


@dataclass(frozen=True)
class Cells:
    """Bank rows grouped by one level's context key, each cell's rows sorted by quality, and that level's prior
    strength `kappa` (0: always trusted)."""

    order: Int[Tensor, " n"]
    keys: Int[Tensor, " c"]
    starts: Int[Tensor, " c"]
    counts: Int[Tensor, " c"]
    kappa: float

    @classmethod
    def of(
        cls,
        keys: Int[Tensor, " n"],
        quality: Float[Tensor, " n"],
        beliefs: Float[Tensor, "n v"],
        parent: Int[Tensor, " n"],
    ) -> "Cells":
        by_quality = quality.argsort()
        order = by_quality[keys[by_quality].argsort(stable=True)]
        unique, counts = torch.unique_consecutive(keys[order], return_counts=True)
        return cls(order, unique, counts.cumsum(0) - counts, counts, Cells.shrinkage(keys, parent, beliefs))

    @staticmethod
    def shrinkage(keys: Int[Tensor, " n"], parent: Int[Tensor, " n"], beliefs: Float[Tensor, "n v"]) -> float:
        """`sigma_within^2 / tau^2`: the within-cell belief variance over the variance of cell means about their
        parent's mean, by the method of moments; 0 when the level is its own parent."""
        if torch.equal(keys, parent):
            return 0.0
        cell, of_row = torch.unique(keys, return_inverse=True)
        _, parent_row = torch.unique(parent, return_inverse=True)
        n = torch.bincount(of_row, minlength=len(cell)).double()
        x = beliefs.double()
        mean = torch.zeros(len(cell), x.shape[1], dtype=x.dtype).index_add_(0, of_row, x) / n[:, None]
        within = float(((x - mean[of_row]) ** 2).sum() / (len(x) - len(cell)))
        parent_n = torch.bincount(parent_row).double()
        parent_mean = torch.zeros(len(parent_n), x.shape[1], dtype=x.dtype).index_add_(0, parent_row, x)
        parent_mean /= parent_n[:, None]
        cell_parent = torch.zeros(len(cell), dtype=torch.long).scatter_(0, of_row, parent_row)
        many = n >= 2  # noqa: PLR2004
        spread = ((mean - parent_mean[cell_parent]) ** 2).sum(1) - within / n
        tau = float(spread[many].mean())
        return within / tau if tau > 0 else math.inf

    def find(self, keys: Int[Tensor, "b w"]) -> tuple[Int[Tensor, "b w"], Int[Tensor, "b w"]]:
        """Each key's cell start and row count, count 0 where no bank row has it."""
        at = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
        found = self.keys[at] == keys
        return self.starts[at], torch.where(found, self.counts[at], 0)


@dataclass(frozen=True)
class Coupling:
    """A Gaussian copula over a span's quality quantiles: correlation `a + b * phi**lag` at each lag."""

    a: float
    b: float
    phi: float

    @classmethod
    def fit(cls, correlation: Float[Tensor, " k"], pairs: Float[Tensor, " k"]) -> "Coupling":
        """The non-negative `(a, b)` and `phi` in (0, 1) minimising the pair-weighted squared error over lags `1..k`;
        a lag with no correlation measured weighs nothing."""
        lags = torch.arange(1, len(correlation) + 1).double()
        measured = correlation.isfinite()
        weight = torch.where(measured, pairs.double(), 0.0)
        target = torch.where(measured, correlation.double(), 0.0)
        best = (math.inf, cls(0.0, 0.0, 0.0))
        for phi in torch.arange(1, GRID).double() / GRID:
            design = torch.stack([torch.ones_like(lags), phi**lags], 1) * weight.sqrt()[:, None]
            solved = torch.linalg.lstsq(design, target * weight.sqrt()).solution.clamp(min=0)
            residual = float((weight * (target - solved[0] - solved[1] * phi**lags) ** 2).sum())
            if residual < best[0] and float(solved.sum()) <= 1:
                best = (residual, cls(float(solved[0]), float(solved[1]), float(phi)))
        return best[1]

    def quantiles(self, shape: tuple[int, int], generator: torch.Generator) -> Float[Tensor, "b w"]:
        """Uniform quantiles with the coupling's correlation along each span."""
        shared = torch.randn(shape[0], 1, generator=generator)
        steps = torch.randn(shape, generator=generator)
        local = torch.empty(shape)
        local[:, 0] = steps[:, 0]
        for slot in range(1, shape[1]):
            local[:, slot] = self.phi * local[:, slot - 1] + math.sqrt(1 - self.phi**2) * steps[:, slot]
        alone = torch.randn(shape, generator=generator)
        z = math.sqrt(self.a) * shared + math.sqrt(self.b) * local + math.sqrt(max(0.0, 1 - self.a - self.b)) * alone
        return torch.special.ndtr(z)


@dataclass(frozen=True)
class Quality:
    """Each real row's quality and its normal score within its own state."""

    real: Real

    def raw(self) -> Float[Tensor, " n"]:
        """Log-probability of the true state; minus the top log-probability for `OOV`."""
        states, beliefs = self.real.states, self.real.beliefs
        true = beliefs.gather(1, states.clamp(max=OOV - 1)[:, None]).squeeze(1)
        return torch.where(states < OOV, true, -beliefs.max(1).values)

    def scores(self) -> Float[Tensor, " n"]:
        """Each in-span row's normal score of its quality rank among its state's in-span rows; NaN off the spans."""
        rows = self.real.index[self.real.index != PAD]
        raw, states = self.raw()[rows], self.real.states[rows]
        by_raw = raw.argsort()
        order = by_raw[states[by_raw].argsort(stable=True)]
        counts = torch.bincount(states, minlength=STATES)
        starts = counts.cumsum(0) - counts
        sorted_states = states[order]
        rank = torch.arange(len(order)) - starts[sorted_states]
        scores = torch.full((len(self.real.states),), torch.nan)
        scores[rows[order]] = torch.special.ndtri((rank + 0.5) / counts[sorted_states])
        return scores

    def lagged(self) -> tuple[Float[Tensor, " k"], Float[Tensor, " k"]]:
        """The correlation of quality scores `lag` rows apart in a span, lags `1..LAGS`, and each lag's pair count."""
        index = self.real.index
        scores = torch.where(self.real.states < OOV, self.scores(), torch.nan)
        z = torch.where(index != PAD, scores[index.clamp(min=0)], torch.nan)
        correlations: list[float] = []
        pairs: list[float] = []
        for lag in range(1, LAGS + 1):
            a, b = z[:, :-lag].flatten(), z[:, lag:].flatten()
            both = ~(a.isnan() | b.isnan())
            enough = int(both.sum()) > 2  # noqa: PLR2004
            correlations.append(float(torch.corrcoef(torch.stack([a[both], b[both]]))[0, 1]) if enough else math.nan)
            pairs.append(float(both.sum()))
        return torch.tensor(correlations), torch.tensor(pairs)


@dataclass(frozen=True)
class Correlated:
    """Draws a span's beliefs with its rows' contexts and coupled quality: a `Drawer` for `Simulated`."""

    beliefs: Float[Tensor, "n v"]
    levels: tuple[Cells, ...]
    coupling: Coupling

    @classmethod
    def of(cls, real: Real) -> "Correlated":
        spans = real.span_states(torch.arange(real.count()))
        inside = real.index != PAD
        rows = real.index[inside]
        keys = [key[inside] for key in Context.of(spans).cells()]
        quality = Quality(real).raw()[rows]
        beliefs = real.beliefs[rows]
        parents = [*keys[1:], keys[-1]]
        levels = tuple(Cells.of(key, quality, beliefs, parent) for key, parent in zip(keys, parents, strict=True))
        if len(levels[-1].keys) < STATES:
            raise ValueError("a state has no in-span row to draw")
        return cls(beliefs, levels, Coupling.fit(*Quality(real).lagged()))

    @classmethod
    def contextual(cls, real: Real) -> "Correlated":
        """Context cells alone, each row's quantile drawn independently: the coupling's instrument arm."""
        return replace(cls.of(real), coupling=Coupling(0.0, 0.0, 0.0))

    @classmethod
    def coupled(cls, real: Real) -> "Correlated":
        """The coupling alone over each row's own state: the context cells' instrument arm."""
        made = cls.of(real)
        return replace(made, levels=made.levels[-1:])

    def draw(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Float[Tensor, "b w v"]:
        """One belief per position, zeros past a span's end."""
        quantile = self.coupling.quantiles((states.shape[0], states.shape[1]), generator)
        chosen = torch.full(states.shape, -1)
        keys = Context.of(states).cells()[-len(self.levels) :]
        for cells, key in zip(self.levels, keys, strict=True):
            start, count = cells.find(key)
            trusted = torch.rand(states.shape, generator=generator) * (count + cells.kappa) < count
            take = (chosen < 0) & trusted & (count > 0)
            offset = (quantile * count).long().clamp(max=(count - 1).clamp(min=0))
            chosen = torch.where(take, cells.order[(start + offset).clamp(max=len(cells.order) - 1)], chosen)
        rows = self.beliefs[chosen.clamp(min=0)]
        return torch.where((states != PAD)[..., None], rows, 0.0)


@dataclass(frozen=True)
class Profile:
    """What a sampler must reproduce, read on spans of beliefs: quality correlation at lags 1-3, and the mean
    log-probability a row gives its own, its right and its left neighbour's state (in-vocabulary states only)."""

    lag1: float
    lag2: float
    lag3: float
    own: float
    right: float
    left: float

    @classmethod
    def of(cls, real: Real) -> "Profile":
        correlation, _ = Quality(real).lagged()
        index = real.index
        lag1, lag2, lag3 = (float(c) for c in correlation[:3])
        return cls(
            lag1,
            lag2,
            lag3,
            Profile.given(real, index, index),
            Profile.given(real, index[:, :-1], index[:, 1:]),
            Profile.given(real, index[:, 1:], index[:, :-1]),
        )

    @staticmethod
    def given(real: Real, rows: Int[Tensor, "g k"], states_of: Int[Tensor, "g k"]) -> float:
        """Mean log-probability row `rows[i]` gives the state of `states_of[i]`, where both exist and it is a word."""
        both = (rows != PAD) & (states_of != PAD)
        row, other = rows[both], real.states[states_of[both]]
        word = other < OOV
        return float(real.beliefs[row[word], other[word]].mean())

    @staticmethod
    def simulated(real: Real, drawer: Drawer, seed: int) -> "Profile":
        """The profile of `real`'s spans with every belief redrawn by `drawer`."""
        states = real.span_states(torch.arange(real.count()))
        beliefs = drawer.draw(states, torch.Generator().manual_seed(seed))
        inside = states != PAD
        index = torch.full(states.shape, PAD)
        index[inside] = torch.arange(int(inside.sum()))
        return Profile.of(Real(beliefs[inside], states[inside], index))

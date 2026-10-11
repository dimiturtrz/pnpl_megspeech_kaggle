import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.training.sampling.curriculum import (
    Annealed,
    Balanced,
    Base,
    Curriculum,
    Natural,
    Observation,
    Prioritised,
    Robust,
    Schedule,
    Stratified,
    Uniform,
)
from megspeech.training.sampling.strata import Strata


def split(subject: list[int], weight: list[float]) -> Split:
    n = len(subject)
    spans = Spans(np.arange(n)[:, None], np.zeros((n, 1)))
    windows = Windows.stacked(torch.zeros(n, 1, 2))
    labels = np.zeros(n, dtype=np.int64)
    return Split.of(windows, (labels, labels), spans, np.array(weight), np.array(subject))


TRAIN = split([0, 0, 1, 2], [1.0, 3.0, 2.0, 5.0])


@pytest.mark.parametrize(("epoch", "epochs", "progress"), [(0, 10, 0.0), (9, 10, 1.0), (3, 7, 0.5), (0, 1, 0.0)])
def test_progress(epoch: int, epochs: int, progress: float) -> None:
    assert Schedule(TRAIN, epoch, epochs).progress == progress


def test_losses() -> None:
    """Each span's labelled words' cross-entropy summed, and their count; ignored positions add neither."""
    target = torch.tensor([[0, 1, IGNORE], [IGNORE, IGNORE, IGNORE]])
    summed, counts = Observation(torch.tensor([0, 1]), torch.zeros(2, 3, 50), target).losses()
    assert summed.tolist() == pytest.approx([2 * math.log(50), 0.0])
    assert counts.tolist() == [2.0, 0.0]


def test_natural_weights() -> None:
    assert Natural().weights(Schedule(TRAIN, 0, 10)).tolist() == [1.0, 3.0, 2.0, 5.0]


def test_uniform_weights() -> None:
    assert Uniform().weights(Schedule(TRAIN, 0, 10)).tolist() == [1.0] * 4


def test_annealed_observe() -> None:
    """Annealing wraps its base: uniform spans keep their equal weights inside a subject, and observing reaches it."""
    assert Annealed(0.9, Uniform()).weights(Schedule(TRAIN, 9, 10))[:2].tolist() == pytest.approx([0.45, 0.45])
    seen = prioritised()
    Annealed(0.9, seen).observe(
        Observation(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    )
    assert seen.seen.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_annealed_of() -> None:
    assert isinstance(Annealed.of(0.0, Natural()), Natural)
    assert Annealed.of(0.9, Natural()) == Annealed(0.9, Natural())


@pytest.mark.parametrize(("epoch", "deep"), [(0, 1 / 3), (9, 0.9), (3, 1 / 3 + (0.9 - 1 / 3) / 3)])
def test_annealed_weights(epoch: int, deep: float) -> None:
    """Subject shares follow the schedule; inside the Deep subject spans keep their 1:3 ratio."""
    weights = Annealed(0.9, Natural()).weights(Schedule(TRAIN, epoch, 10))
    shares = weights / weights.sum()
    assert float(shares[:2].sum()) == pytest.approx(deep)
    assert float(shares[2]) == pytest.approx((1 - deep) / 2)
    assert float(shares[1] / shares[0]) == pytest.approx(3.0)


def test_annealed_weights_single_subject() -> None:
    lone = split([0, 0], [1.0, 3.0])
    weights = Annealed(0.9, Natural()).weights(Schedule(lone, 5, 10))
    assert torch.allclose(weights / weights.sum(), torch.tensor([0.25, 0.75]))


SPANNED = Split.of(
    Windows.stacked(torch.zeros(6, 1, 2)),
    (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)),
    Spans(np.array([[0, 1, 2], [3, -1, -1], [4, 5, -1]]), np.array([[0, 0.2, 0.4], [0, 0, 0], [0, 1.0, 0]])),
    np.ones(6),
)
"""Spans of 3 words at 5 words/s, 1 word (rate 0), 2 words at 1 word/s: each the top, bottom and middle tercile."""


def robust(strata: list[int], power: float = 1.0) -> Robust:
    zeros = torch.zeros(max(strata) + 1)
    return Robust(Natural(), torch.tensor(strata), power, torch.ones_like(zeros), zeros.clone(), zeros.clone())


def test_robust_of() -> None:
    assert isinstance(Robust.of(Natural(), SPANNED, 0.0), Natural)
    made = Robust.of(Natural(), SPANNED, 2.0)
    assert isinstance(made, Robust)
    assert made.strata.tolist() == [8, 0, 4]
    assert made.tilt.tolist() == [1.0] * 9


def test_robust_observe() -> None:
    """Each labelled word's cross-entropy lands in its span's stratum; ignored positions add neither loss nor count."""
    Natural().observe(Observation(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64)))
    Annealed(0.9, Natural()).observe(
        Observation(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    )
    made = robust([1, 0, 1])
    target = torch.tensor([[0, 1, IGNORE], [3, IGNORE, IGNORE]])
    made.observe(Observation(torch.tensor([0, 2]), torch.zeros(2, 3, 50), target))
    assert made.counts.tolist() == [0.0, 3.0]
    assert made.sums.tolist() == pytest.approx([0.0, 3 * math.log(50)])


def test_robust_weights() -> None:
    """A stratum above the mean loss is drawn more by its ratio to the power, one below never less; the losses are
    spent, and an epoch observing nothing keeps the tilt."""
    made = robust([0, 1, 0, 1], power=2.0)
    made.sums += torch.tensor([4.0, 1.0])
    made.counts += torch.tensor([1.0, 1.0])
    expected = [1.6**2, 3.0, 2 * 1.6**2, 5.0]
    assert made.weights(Schedule(TRAIN, 1, 10)).tolist() == pytest.approx(expected)
    assert made.counts.tolist() == [0.0, 0.0]
    assert made.weights(Schedule(TRAIN, 2, 10)).tolist() == pytest.approx(expected)


def prioritised(power: float = 1.0, visibility: float = 1.0, capped: list[bool] | None = None) -> Prioritised:
    made = Prioritised.of(Natural(), TRAIN, power, visibility, None if capped is None else torch.tensor(capped))
    assert isinstance(made, Prioritised)
    return made


def test_prioritised_of() -> None:
    assert isinstance(Prioritised.of(Natural(), TRAIN, 0.0, 0.0), Natural)
    made = prioritised()
    assert made.seen.tolist() == [0.0] * 4
    assert made.capped.tolist() == [False] * 4


def test_prioritised_observe() -> None:
    """Draws are counted, difficulty moves halfway to each new loss, and a span with no labelled word keeps its own."""
    made = prioritised()
    made.observe(Observation(torch.tensor([0, 1]), torch.zeros(2, 1, 50), torch.tensor([[0], [IGNORE]])))
    assert made.seen.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert made.difficulty.tolist() == pytest.approx([math.log(50), 0.0, 0.0, 0.0])
    logits = torch.zeros(1, 1, 50)
    logits[0, 0, 0] = 1e4
    made.observe(Observation(torch.tensor([0]), logits, torch.tensor([[0]])))
    assert made.difficulty[0].item() == pytest.approx(math.log(50) / 2)


def test_prioritised_weights() -> None:
    made = prioritised(power=0.0)
    made.observe(Observation(torch.tensor([0, 1]), torch.zeros(2, 1, 50), torch.zeros(2, 1, dtype=torch.int64)))
    weights = made.weights(Schedule(TRAIN, 0, 10))
    assert weights[:2].tolist() == pytest.approx([0.5, 1.5])
    assert weights[2:].tolist() == pytest.approx([2.0, 5.0])


def test_prioritised_weights_harder_beats_easier() -> None:
    """Unseen spans take the highest difficulty, so they and the harder seen span draw above the easier one."""
    made = prioritised(visibility=0.0)
    made.difficulty = torch.tensor([1.0, 3.0, 0.0, 0.0])
    mean = (1.0 + 3.0 + 3.0 + 3.0) / 4
    expected = [1.0 / mean, 3.0 * 3.0 / mean, 2.0 * 3.0 / mean, 5.0 * 3.0 / mean]
    assert made.weights(Schedule(TRAIN, 0, 10)).tolist() == pytest.approx(expected)


def test_prioritised_weights_nothing_observed_is_base() -> None:
    assert prioritised(power=2.0, visibility=0.0).weights(Schedule(TRAIN, 0, 10)).tolist() == [1.0, 3.0, 2.0, 5.0]


def test_prioritised_weights_floor() -> None:
    made = prioritised(power=1.0, visibility=3.0)
    made.seen = torch.tensor([100.0, 0.0, 0.0, 0.0])
    made.difficulty = torch.tensor([0.1, 5.0, 5.0, 5.0])
    assert made.weights(Schedule(TRAIN, 0, 10))[0].item() == pytest.approx(0.25)


def test_prioritised_weights_capped() -> None:
    """A capped span never exceeds its base weight however hard; an uncapped one does."""
    made = prioritised(visibility=0.0, capped=[False, False, True, False])
    made.difficulty = torch.tensor([1.0, 9.0, 9.0, 1.0])
    weights = made.weights(Schedule(TRAIN, 0, 10))
    assert weights[1].item() > 3.0
    assert weights[2].item() == pytest.approx(2.0)


def test_prioritised_weights_negative_power_and_ceiling() -> None:
    """A negative power ranks the easier span above the harder; none rises past four times its base weight."""
    made = prioritised(power=-1.0, visibility=0.0)
    made.difficulty = torch.tensor([1.0, 3.0, 1.0, 1.0])
    weights = made.weights(Schedule(TRAIN, 0, 10))
    assert weights[0].item() > weights[1].item() / 3.0
    boosted = prioritised(power=5.0, visibility=0.0)
    boosted.difficulty = torch.tensor([1.0, 1.0, 1.0, 100.0])
    assert boosted.weights(Schedule(TRAIN, 0, 10))[3].item() == pytest.approx(20.0)


def test_curriculum_summary() -> None:
    assert Natural().summary(Schedule(TRAIN, 0, 10), Natural().weights(Schedule(TRAIN, 0, 10))) == {}


def test_annealed_summary() -> None:
    made = Annealed(0.9, Natural())
    assert made.summary(Schedule(TRAIN, 0, 10), made.weights(Schedule(TRAIN, 0, 10))) == {}


def test_robust_summary() -> None:
    made = robust([0, 1, 0, 1])
    assert made.summary(Schedule(TRAIN, 0, 10), made.weights(Schedule(TRAIN, 0, 10))) == {}


def test_prioritised_summary() -> None:
    """A prioritised one reports the ratio quantiles, clamp shares and ESS."""
    made = prioritised(power=1.0, visibility=0.0)
    flat = made.summary(Schedule(TRAIN, 0, 10), made.weights(Schedule(TRAIN, 0, 10)))
    assert flat["ratio_p50"] == pytest.approx(1.0)
    assert flat["at_floor"] == 0.0
    assert flat["ess"] == pytest.approx((11.0**2 / (1 + 9 + 4 + 25)) / 4)
    made.seen = torch.tensor([100.0, 0.0, 0.0, 0.0])
    tilted = prioritised(power=0.0, visibility=3.0)
    tilted.seen = made.seen
    out = tilted.summary(Schedule(TRAIN, 0, 10), tilted.weights(Schedule(TRAIN, 0, 10)))
    assert out["at_floor"] == pytest.approx(0.25)
    assert out["at_ceiling"] == 0.0
    assert out["ratio_p95"] == pytest.approx(1.0)


def test_stratified_of() -> None:
    made = Stratified.of(0.5, TRAIN_STRATA)
    assert made.alpha == 0.5
    assert made.strata.tolist() == Strata(TRAIN_STRATA).sourced().tolist()


def test_stratified_weights() -> None:
    """At alpha 1 each stratum's spans sum to the same total whatever its size; at 0 every span draws alike."""
    made = Stratified.of(1.0, TRAIN_STRATA)
    assert made.strata.tolist() == Strata(TRAIN_STRATA).sourced().tolist()
    weights = made.weights(Schedule(TRAIN_STRATA, 0, 10))
    totals = torch.zeros(int(made.strata.max()) + 1).index_add_(0, made.strata, weights)
    assert totals.tolist() == pytest.approx([1.0] * len(totals))
    assert Stratified.of(0.0, TRAIN_STRATA).weights(Schedule(TRAIN_STRATA, 0, 10)).tolist() == [1.0] * 3


def test_stratified_summary() -> None:
    made = Stratified.of(1.0, TRAIN_STRATA)
    made.observe(Observation(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64)))
    assert made.summary(Schedule(TRAIN_STRATA, 0, 10), made.weights(Schedule(TRAIN_STRATA, 0, 10))) == {}


def test_base_of() -> None:
    assert Base.UNIFORM.of(TRAIN, 1.0) == Uniform()
    assert Base.NATURAL.of(TRAIN, 1.0) == Natural()
    stratified = Base.STRATIFIED.of(TRAIN_STRATA, 0.5)
    assert isinstance(stratified, Stratified)
    assert stratified.alpha == 0.5
    assert stratified.strata.tolist() == Strata(TRAIN_STRATA).sourced().tolist()


def test_uniform_summary() -> None:
    Uniform().observe(Observation(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64)))
    assert Uniform().summary(Schedule(TRAIN, 0, 10), Uniform().weights(Schedule(TRAIN, 0, 10))) == {}


def test_robust_and_prioritised_both_observe() -> None:
    """Prioritised over Robust: one `observe` reaches both, so each tilt draws by the loss it saw."""
    inner = robust([0, 1, 0, 1])
    outer = Prioritised.of(inner, TRAIN, 1.0, 1.0)
    assert isinstance(outer, Prioritised)
    outer.observe(Observation(torch.tensor([0, 1]), torch.zeros(2, 1, 50), torch.zeros(2, 1, dtype=torch.int64)))
    assert outer.seen.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert inner.counts.tolist() == [1.0, 1.0]
    assert outer.weights(Schedule(TRAIN, 1, 10)).shape == (4,)
    assert outer.summary(Schedule(TRAIN, 1, 10), outer.weights(Schedule(TRAIN, 1, 10)))["ess"] > 0


TRAIN_STRATA = replace(SPANNED, source=torch.tensor([0, 0, 0, 1, 1, 1]))


LABELLED = Split.of(
    Windows.stacked(torch.zeros(7, 1, 2)),
    (np.array([0, 0, 0, 0, 1, IGNORE, 0]), np.zeros(7, dtype=np.int64)),
    Spans(np.array([[0, 1], [2, 3], [4, 6], [5, -1]]), np.zeros((4, 2))),
    np.ones(7),
)


def balanced(power: float, base: Curriculum | None = None) -> Balanced:
    made = Balanced.of(Natural() if base is None else base, LABELLED, power)
    assert isinstance(made, Balanced)
    return made


def test_balanced_of() -> None:
    """At 0 the base itself; label 1 is 1 of 6 labelled words, label 0 is 5, so at power 1 its words weigh 5 to 1:
    0.6 and 3 once normalised to a mean of 1 over the six, and a span with no labelled word keeps 1."""
    assert Balanced.of(Natural(), LABELLED, 0.0) == Natural()
    made = balanced(1.0)
    assert made.rarity.tolist() == pytest.approx([0.6, 0.6, 1.8, 1.0])
    assert made.rare.tolist() == [False, False, True, False]


def test_balanced_labels() -> None:
    labels, valid = Balanced.labels(LABELLED)
    assert labels.tolist() == [[0, 0], [0, 0], [1, 0], [0, 0]]
    assert valid.tolist() == [[True, True], [True, True], [True, True], [False, False]]


def test_balanced_weights() -> None:
    """The base's weights times each span's rarity: the rare word's span draws most."""
    made = balanced(0.5)
    weights = made.weights(Schedule(LABELLED, 0, 10))
    assert torch.equal(weights, Natural().weights(Schedule(LABELLED, 0, 10)) * made.rarity)
    assert int(weights.argmax()) == 2


def test_balanced_observe() -> None:
    base = prioritised(power=1.0, visibility=0.0)
    balanced(1.0, base).observe(
        Observation(torch.tensor([3]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    )
    assert base.seen.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_balanced_summary() -> None:
    """The rare span's share of the draw rises above its share of the base."""
    made = balanced(1.0)
    out = made.summary(Schedule(LABELLED, 0, 10), made.weights(Schedule(LABELLED, 0, 10)))
    assert out["rare_share_base"] == pytest.approx(0.25)
    assert out["rare_share"] > out["rare_share_base"]


def test_balanced_share() -> None:
    assert balanced(1.0).share(torch.tensor([1.0, 1.0, 2.0, 0.0])) == pytest.approx(0.5)

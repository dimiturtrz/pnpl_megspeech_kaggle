import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.training.sampling.curriculum import Annealed, Natural, Prioritised, Robust


def split(subject: list[int], weight: list[float]) -> Split:
    n = len(subject)
    spans = Spans(np.arange(n)[:, None], np.zeros((n, 1)))
    windows = Windows.stacked(torch.zeros(n, 1, 2))
    labels = np.zeros(n, dtype=np.int64)
    return Split.of(windows, (labels, labels), spans, np.array(weight), np.array(subject))


TRAIN = split([0, 0, 1, 2], [1.0, 3.0, 2.0, 5.0])


def test_natural_weights() -> None:
    assert Natural().weights(TRAIN, 0, 10).tolist() == [1.0, 3.0, 2.0, 5.0]


def test_annealed_of() -> None:
    assert isinstance(Annealed.of(0.0), Natural)
    assert Annealed.of(0.9) == Annealed(0.9)


@pytest.mark.parametrize(("epoch", "deep"), [(0, 1 / 3), (9, 0.9), (3, 1 / 3 + (0.9 - 1 / 3) / 3)])
def test_annealed_weights(epoch: int, deep: float) -> None:
    """Subject shares follow the schedule; inside the Deep subject spans keep their 1:3 ratio."""
    weights = Annealed(0.9).weights(TRAIN, epoch, 10)
    shares = weights / weights.sum()
    assert float(shares[:2].sum()) == pytest.approx(deep)
    assert float(shares[2]) == pytest.approx((1 - deep) / 2)
    assert float(shares[1] / shares[0]) == pytest.approx(3.0)


def test_annealed_weights_single_subject() -> None:
    lone = split([0, 0], [1.0, 3.0])
    weights = Annealed(0.9).weights(lone, 5, 10)
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


def test_stratify() -> None:
    """Subject, then length tercile, then rate tercile; a second subject's strata sit past the first's."""
    assert Robust.stratify(SPANNED).tolist() == [8, 0, 4]
    other = replace(SPANNED, subject=torch.ones(6, dtype=torch.int64))
    assert Robust.stratify(other).tolist() == [17, 9, 13]


def test_binned() -> None:
    assert Robust.binned(torch.tensor([3.0, 1.0, 2.0])).tolist() == [2, 0, 1]
    assert Robust.binned(torch.arange(6.0)).tolist() == [0, 0, 1, 1, 2, 2]


def test_observe() -> None:
    """Each labelled word's cross-entropy lands in its span's stratum; ignored positions add neither loss nor count."""
    Natural().observe(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    Annealed(0.9).observe(torch.tensor([0]), torch.zeros(1, 1, 50), torch.zeros(1, 1, dtype=torch.int64))
    made = robust([1, 0, 1])
    target = torch.tensor([[0, 1, IGNORE], [3, IGNORE, IGNORE]])
    made.observe(torch.tensor([0, 2]), torch.zeros(2, 3, 50), target)
    assert made.counts.tolist() == [0.0, 3.0]
    assert made.sums.tolist() == pytest.approx([0.0, 3 * math.log(50)])


def test_robust_weights() -> None:
    """A stratum above the mean loss is drawn more by its ratio to the power, one below never less; the losses are
    spent, and an epoch observing nothing keeps the tilt."""
    made = robust([0, 1, 0, 1], power=2.0)
    made.sums += torch.tensor([4.0, 1.0])
    made.counts += torch.tensor([1.0, 1.0])
    expected = [1.6**2, 3.0, 2 * 1.6**2, 5.0]
    assert made.weights(TRAIN, 1, 10).tolist() == pytest.approx(expected)
    assert made.counts.tolist() == [0.0, 0.0]
    assert made.weights(TRAIN, 2, 10).tolist() == pytest.approx(expected)


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
    made.observe(torch.tensor([0, 1]), torch.zeros(2, 1, 50), torch.tensor([[0], [IGNORE]]))
    assert made.seen.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert made.difficulty.tolist() == pytest.approx([math.log(50), 0.0, 0.0, 0.0])
    logits = torch.zeros(1, 1, 50)
    logits[0, 0, 0] = 1e4
    made.observe(torch.tensor([0]), logits, torch.tensor([[0]]))
    assert made.difficulty[0].item() == pytest.approx(math.log(50) / 2)


def test_prioritised_weights_unseen_beats_seen() -> None:
    made = prioritised(power=0.0)
    made.observe(torch.tensor([0, 1]), torch.zeros(2, 1, 50), torch.zeros(2, 1, dtype=torch.int64))
    weights = made.weights(TRAIN, 0, 10)
    assert weights[:2].tolist() == pytest.approx([0.5, 1.5])
    assert weights[2:].tolist() == pytest.approx([2.0, 5.0])


def test_prioritised_weights_harder_beats_easier() -> None:
    """Unseen spans take the highest difficulty, so they and the harder seen span draw above the easier one."""
    made = prioritised(visibility=0.0)
    made.difficulty = torch.tensor([1.0, 3.0, 0.0, 0.0])
    mean = (1.0 + 3.0 + 3.0 + 3.0) / 4
    expected = [1.0 / mean, 3.0 * 3.0 / mean, 2.0 * 3.0 / mean, 5.0 * 3.0 / mean]
    assert made.weights(TRAIN, 0, 10).tolist() == pytest.approx(expected)


def test_prioritised_weights_nothing_observed_is_base() -> None:
    assert prioritised(power=2.0, visibility=0.0).weights(TRAIN, 0, 10).tolist() == [1.0, 3.0, 2.0, 5.0]


def test_prioritised_weights_floor() -> None:
    made = prioritised(power=1.0, visibility=3.0)
    made.seen = torch.tensor([100.0, 0.0, 0.0, 0.0])
    made.difficulty = torch.tensor([0.1, 5.0, 5.0, 5.0])
    assert made.weights(TRAIN, 0, 10)[0].item() == pytest.approx(0.25)


def test_prioritised_weights_capped() -> None:
    """A capped span never exceeds its base weight however hard; an uncapped one does."""
    made = prioritised(visibility=0.0, capped=[False, False, True, False])
    made.difficulty = torch.tensor([1.0, 9.0, 9.0, 1.0])
    weights = made.weights(TRAIN, 0, 10)
    assert weights[1].item() > 3.0
    assert weights[2].item() == pytest.approx(2.0)


def test_prioritised_weights_negative_power_and_ceiling() -> None:
    """A negative power ranks the easier span above the harder; none rises past four times its base weight."""
    made = prioritised(power=-1.0, visibility=0.0)
    made.difficulty = torch.tensor([1.0, 3.0, 1.0, 1.0])
    weights = made.weights(TRAIN, 0, 10)
    assert weights[0].item() > weights[1].item() / 3.0
    boosted = prioritised(power=5.0, visibility=0.0)
    boosted.difficulty = torch.tensor([1.0, 1.0, 1.0, 100.0])
    assert boosted.weights(TRAIN, 0, 10)[3].item() == pytest.approx(20.0)


def test_summary() -> None:
    """Untilted curricula report nothing; a prioritised one reports the ratio quantiles, clamp shares and ESS."""
    for plain in (Natural(), Annealed(0.9), robust([0, 1, 0, 1])):
        assert plain.summary(TRAIN, 0, 10, plain.weights(TRAIN, 0, 10)) == {}
    made = prioritised(power=1.0, visibility=0.0)
    flat = made.summary(TRAIN, 0, 10, made.weights(TRAIN, 0, 10))
    assert flat["ratio_p50"] == pytest.approx(1.0)
    assert flat["at_floor"] == 0.0
    assert flat["ess"] == pytest.approx((11.0**2 / (1 + 9 + 4 + 25)) / 4)
    made.seen = torch.tensor([100.0, 0.0, 0.0, 0.0])
    tilted = prioritised(power=0.0, visibility=3.0)
    tilted.seen = made.seen
    out = tilted.summary(TRAIN, 0, 10, tilted.weights(TRAIN, 0, 10))
    assert out["at_floor"] == pytest.approx(0.25)
    assert out["at_ceiling"] == 0.0
    assert out["ratio_p95"] == pytest.approx(1.0)

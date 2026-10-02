import numpy as np
import pytest
import torch

from megspeech.ruler import Coverage, Interval, Ruler
from megspeech.score import BalancedTopK
from megspeech.wordnet import WORDS


def scored(n: int = 200, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, WORDS, generator=generator), torch.randint(WORDS, (n,), generator=generator)


def test_hits() -> None:
    scores = torch.zeros(2, WORDS)
    scores[0, 3] = 1
    scores[1, :20] = 1
    hits = Ruler(k=10).hits(scores, torch.tensor([3, 40]))
    assert hits.tolist() == [True, False]
    assert not Ruler().hits(torch.full((1, WORDS), torch.nan), torch.tensor([0])).any()


def test_balanced() -> None:
    scores, labels = scored()
    ruler = Ruler()
    expected = BalancedTopK()(scores.numpy(), labels.numpy())
    assert ruler.balanced(ruler.hits(scores, labels), labels) == pytest.approx(expected)


def test_draws() -> None:
    scores, labels = scored()
    ruler = Ruler(resamples=300)
    hits = ruler.hits(scores, labels)
    draws = ruler.draws(hits, labels)
    assert draws.shape == (300,)
    assert torch.equal(draws, ruler.draws(hits, labels))
    assert abs(float(draws.mean()) - ruler.balanced(hits, labels)) < 0.03


def test_interval() -> None:
    interval = Ruler(confidence=0.9).interval(torch.linspace(0, 1, 101))
    assert isinstance(interval, Interval)
    assert (interval.low, interval.high) == pytest.approx((0.05, 0.95))


def test_score() -> None:
    hits = torch.tensor([[True, False, True, True], [False, False, False, False]])
    labels = torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2]])
    assert Ruler.score(hits, labels).tolist() == pytest.approx([0.75, 0.0])


def test_of() -> None:
    coverage = Coverage.of(torch.tensor(np.r_[[0] * 10, [1] * 2, [2] * 6]))
    assert coverage == Coverage(3, 6.0, 1)
    assert Coverage.of(torch.tensor([], dtype=torch.long)) == Coverage(0, 0.0, 0)

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.ruler import Coverage, Graded, Interval, Ruler, main

CPU = torch.device("cpu")


def scored(n: int = 200, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, VOCAB_SIZE, generator=generator), torch.randint(VOCAB_SIZE, (n,), generator=generator)


def graded(seed: int = 0) -> Graded:
    ruler = Ruler()
    domains = {}
    for name, n in (("a", 120), ("b", 80)):
        scores, labels = scored(n, seed=n)
        domains[name] = (ruler.hits(scores + seed * torch.nn.functional.one_hot(labels, VOCAB_SIZE), labels), labels)
    return Graded(domains)


def test_hits() -> None:
    scores = torch.zeros(2, VOCAB_SIZE)
    scores[0, 3] = 1
    scores[1, :20] = 1
    hits = Ruler(k=10).hits(scores, torch.tensor([3, 40]))
    assert hits.tolist() == [True, False]
    assert not Ruler().hits(torch.full((1, VOCAB_SIZE), torch.nan), torch.tensor([0])).any()


def test_balanced() -> None:
    scores, labels = scored()
    ruler = Ruler()
    rank = (scores.numpy() > np.take_along_axis(scores.numpy(), labels.numpy()[:, None], 1)).sum(1)
    expected = np.mean([(rank[labels.numpy() == c] < ruler.k).mean() for c in np.unique(labels.numpy())])
    assert ruler.balanced(ruler.hits(scores, labels), labels) == pytest.approx(expected)


def test_draws() -> None:
    scores, labels = scored()
    ruler = Ruler(resamples=300)
    hits = ruler.hits(scores, labels)
    draws = ruler.draws(hits, labels)
    assert draws.shape == (300,)
    assert torch.equal(draws, ruler.draws(hits, labels))
    assert abs(float(draws.mean()) - ruler.balanced(hits, labels)) < 0.03


def test_point() -> None:
    ruler = Ruler()
    runs = graded()
    expected = np.mean([ruler.balanced(hits, labels) for hits, labels in runs.domains.values()])
    assert ruler.point(runs) == pytest.approx(expected)


def test_macro() -> None:
    ruler = Ruler(resamples=200)
    draws = ruler.macro(graded())
    assert draws.shape == (200,)
    assert abs(float(draws.mean()) - ruler.point(graded())) < 0.03
    assert ruler.macro(Graded({})).isnan().all()


def test_delta() -> None:
    ruler = Ruler(resamples=200)
    assert torch.equal(ruler.delta(graded(), graded()), torch.zeros(200))
    assert float(ruler.delta(graded(seed=1), graded()).mean()) > 0
    with pytest.raises(ValueError, match="different words"):
        ruler.delta(graded(), Graded({"a": graded().domains["a"]}))


def test_interval() -> None:
    interval = Ruler(confidence=0.9).interval(torch.linspace(0, 1, 101))
    assert isinstance(interval, Interval)
    assert (interval.low, interval.high) == pytest.approx((0.05, 0.95))


def test_information() -> None:
    labels, uniform = torch.arange(4).repeat(5), torch.ones(VOCAB_SIZE)
    assert Ruler.information(labels, labels, uniform) == pytest.approx(2.0)
    assert Ruler.information(torch.zeros_like(labels), labels, uniform) == pytest.approx(0.0)
    skewed = torch.ones(VOCAB_SIZE)
    skewed[0] = 3
    pair = torch.tensor([0, 1, 1])
    assert Ruler.information(pair, pair, skewed) == pytest.approx(-(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)))


def test_score() -> None:
    hits = torch.tensor([[True, False, True, True], [False, False, False, False]])
    labels = torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2]])
    assert Ruler.score(hits, labels).tolist() == pytest.approx([0.75, 0.0])


def test_of() -> None:
    coverage = Coverage.of(torch.tensor(np.r_[[0] * 10, [1] * 2, [2] * 6]))
    assert coverage == Coverage(3, 6.0, 1)
    assert Coverage.of(torch.tensor([], dtype=torch.long)) == Coverage(0, 0.0, 0)


def test_save(tmp_path: Path) -> None:
    path = graded().save(tmp_path / "hits.pt")
    assert set(torch.load(path, weights_only=True)) == {"a", "b"}


def test_load(tmp_path: Path) -> None:
    runs = graded()
    loaded = Graded.load(runs.save(tmp_path / "hits.pt"), CPU)
    assert loaded.domains.keys() == runs.domains.keys()
    assert all(torch.equal(loaded.domains[n][0], runs.domains[n][0]) for n in runs.domains)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    a, b = graded(seed=1).save(tmp_path / "a.pt"), graded().save(tmp_path / "b.pt")
    monkeypatch.setattr(sys, "argv", ["ruler", str(a), str(b)])
    with caplog.at_level("INFO"):
        main()
    assert "P(a>b)" in caplog.text

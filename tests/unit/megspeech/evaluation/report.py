from typing import override

import numpy as np
import pytest
import torch
from torch import nn

from megspeech.data.runs import CHANNELS
from megspeech.data.spans import Spans
from megspeech.data.split import DENSE, Split
from megspeech.data.vocabulary import WORDS
from megspeech.evaluation.balance import Sinkhorn
from megspeech.evaluation.report import ADJUSTMENTS, Evaluator, Head, Score

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


class Constant(nn.Module):
    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(*slot.shape, WORDS)
        logits[..., 0] = 1
        return logits, logits


def test_logits() -> None:
    out = Evaluator(CPU).logits(Constant(), split())
    assert out[Head.CONTEXT].shape == (12, WORDS)
    assert torch.all(out[Head.WORD].argmax(dim=1) == 0)


def test_evaluate() -> None:
    logits = torch.zeros(3, WORDS)
    logits[:, 0] = 1
    result = Evaluator(CPU).evaluate(logits, torch.tensor([0, 0, 0]), torch.zeros(WORDS))
    assert result.bacc10 == pytest.approx(1.0)
    assert result.n == 3


def test_adjusted() -> None:
    """Unbalanced, `tau` subtracts that share of the log prior; balanced, every class ends with equal mass."""
    logits, log_prior = torch.zeros(4, WORDS), torch.linspace(-1, 0, WORDS)
    logits[:, 0] = 3
    assert torch.allclose(Evaluator(CPU).adjusted(logits, log_prior, 0.5), logits - 0.5 * log_prior)
    mass = Evaluator(CPU, balance=Sinkhorn(50)).adjusted(logits, log_prior, 1.0).log_softmax(-1).logsumexp(0)
    assert torch.allclose(mass, mass.mean().expand_as(mass), atol=1e-3)


def test_best() -> None:
    score = Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7)
    assert (score.tau, score.bacc10, score.n) == (0.5, 0.4, 7)


def test_mean() -> None:
    """One adjustment is chosen for the unweighted mean, not each score's own best."""
    scores = [Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7), Score.best({0.0: 0.6, 0.5: 0.4, 1.0: 0.1}, 3)]
    macro = Score.mean(iter(scores))
    assert (macro.tau, macro.n) == (0.0, 10)
    assert macro.bacc10 == pytest.approx(0.4)
    assert macro.by_tau == pytest.approx({0.0: 0.4, 0.5: 0.4, 1.0: 0.2})


def test_assess() -> None:
    report = Evaluator(CPU).assess(Constant(), split(), torch.zeros(WORDS))
    assert report.dense.n == DENSE
    assert report.novel.n == DENSE
    assert report.context.n == len(split().y)
    assert report.coverage.present == len(set(split().y[split().novel(frozenset())].tolist()))
    assert set(report.top1) == set(report.bits) == set(ADJUSTMENTS)
    assert report.top1[0.0] == pytest.approx(1 / report.coverage.present)
    assert report.bits[0.0] == pytest.approx(0.0)


def test_graded() -> None:
    domains = {"a": split(), "b": split(n=8, seed=1)}
    sizes = {name: len(s.novel(frozenset())) for name, s in domains.items()}
    graded = Evaluator(CPU, min_novel=max(sizes.values())).graded(Constant(), domains, torch.zeros(WORDS), 0.0)
    assert set(graded.domains) == {name for name, size in sizes.items() if size == max(sizes.values())}
    hits, labels = next(iter(graded.domains.values()))
    assert hits.shape == labels.shape == (max(sizes.values()),)
    assert hits.all()


def test_scored() -> None:
    evaluator, domain = Evaluator(CPU), split()
    logits = evaluator.logits(Constant(), domain)
    assert evaluator.scored(logits, domain, torch.zeros(WORDS)) == evaluator.assess(
        Constant(), domain, torch.zeros(WORDS)
    )


def test_grade() -> None:
    evaluator, domains = Evaluator(CPU, min_novel=1), {"a": split()}
    context = {"a": evaluator.logits(Constant(), domains["a"])[Head.CONTEXT]}
    graded = evaluator.grade(context, domains, torch.zeros(WORDS), 0.0)
    expected = evaluator.graded(Constant(), domains, torch.zeros(WORDS), 0.0)
    assert all(torch.equal(graded.domains["a"][i], expected.domains["a"][i]) for i in range(2))


class Counting(Constant):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return super().forward(x, onset, slot, subject)


def test_report() -> None:
    model = Counting()
    Evaluator(CPU, min_novel=1).report(model, {"a": split()}, torch.zeros(WORDS), 0, 0.0)
    once = model.calls
    Evaluator(CPU).logits(model, split())
    assert model.calls == 2 * once
    domains = {"a": split(), "b": split(n=8, seed=1)}
    report = Evaluator(CPU, min_novel=1).report(Constant(), domains, torch.zeros(WORDS), 3, 0.5)
    assert report.epoch == 3
    assert set(report.domains) == {"a", "b"}
    counted = list(report.domains.values())
    assert report.macro.n == sum(r.novel.n for r in counted)
    mean = np.mean([r.novel.by_tau[0.0] for r in counted])
    assert report.macro.by_tau[ADJUSTMENTS[0]] == pytest.approx(mean)
    assert report.interval.low <= report.macro.by_tau[report.macro.tau] <= report.interval.high
    assert report.top1 == pytest.approx(np.mean([r.top1[report.macro.tau] for r in counted]))
    assert report.bits == pytest.approx(0.0)

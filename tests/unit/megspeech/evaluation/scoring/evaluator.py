from typing import override

import numpy as np
import pytest
import torch
from torch import nn

from megspeech.data.batching.split import DENSE, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import ADJUSTMENTS, Head, Score
from megspeech.model.words.layers.norm import MaskedBatchNorm1d

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
        logits = torch.zeros(*slot.shape, VOCAB_SIZE)
        logits[..., 0] = 1
        return logits, logits


def test_logits() -> None:
    out = Evaluator(CPU).logits(Constant(), split())
    assert out[Head.CONTEXT].shape == (12, VOCAB_SIZE)
    assert torch.all(out[Head.WORD].argmax(dim=1) == 0)
    assert torch.allclose(out[Head.WORD].logsumexp(-1), torch.zeros(12), atol=1e-5)
    shifted = Evaluator(CPU, shifts=(-1, 0, 1)).logits(Constant(), split())
    assert torch.allclose(shifted[Head.CONTEXT], out[Head.CONTEXT], atol=1e-5)


class Normed(Constant):
    def __init__(self) -> None:
        super().__init__()
        self.norm = MaskedBatchNorm1d(CHANNELS)

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.norm(x)
        return super().forward(x, onset, slot, subject)


def test_adapt() -> None:
    model = Normed()
    adapted = Evaluator(CPU).adapt(model, split())
    assert isinstance(adapted, Normed)
    assert not adapted.training
    assert int(adapted.norm.num_batches_tracked) > 0
    assert not torch.equal(adapted.norm.running_mean, torch.zeros(CHANNELS))
    assert int(model.norm.num_batches_tracked) == 0
    assert torch.equal(model.norm.running_mean, torch.zeros(CHANNELS))
    out = Evaluator(CPU, adapted=True).logits(model, split())
    assert torch.all(out[Head.CONTEXT].argmax(dim=1) == 0)


def test_sweep() -> None:
    model = Normed().train()
    out = Evaluator(CPU).sweep(model, split())
    assert model.training
    assert int(model.norm.num_batches_tracked) > 0
    assert out[Head.WORD].shape == (12, VOCAB_SIZE)


def test_read() -> None:
    out = Evaluator(CPU).read(Constant(), split())
    assert out[Head.CONTEXT].shape == (12, VOCAB_SIZE)
    assert torch.all(out[Head.CONTEXT].argmax(dim=1) == 0)


def test_evaluate() -> None:
    logits = torch.zeros(3, VOCAB_SIZE)
    logits[:, 0] = 1
    result = Evaluator(CPU).evaluate(logits, torch.tensor([0, 0, 0]), torch.zeros(VOCAB_SIZE))
    assert result.bacc10 == pytest.approx(1.0)
    assert result.n == 3


def test_adjusted() -> None:
    """Unbalanced, `tau` subtracts that share of the log prior; balanced, every class ends with equal mass."""
    logits, log_prior = torch.zeros(4, VOCAB_SIZE), torch.linspace(-1, 0, VOCAB_SIZE)
    logits[:, 0] = 3
    assert torch.allclose(Evaluator(CPU).adjusted(logits, log_prior, 0.5), logits - 0.5 * log_prior)
    mass = Evaluator(CPU, balance=Sinkhorn(50)).adjusted(logits, log_prior, 1.0).log_softmax(-1).logsumexp(0)
    assert torch.allclose(mass, mass.mean().expand_as(mass), atol=1e-3)


def test_unseen() -> None:
    evaluator, domain = Evaluator(CPU), split()
    score = evaluator.unseen(evaluator.logits(Constant(), domain)[Head.CONTEXT], domain, torch.zeros(VOCAB_SIZE))
    assert score.n == len(domain.novel(frozenset()))


def test_counted() -> None:
    scores = {"few": Score.best({0.0: 0.2}, 4), "enough": Score.best({0.0: 0.4}, 5)}
    assert list(Evaluator(CPU, min_novel=5).counted(scores)) == ["enough"]


def test_scored() -> None:
    evaluator, domain = Evaluator(CPU), split()
    report = evaluator.scored(evaluator.logits(Constant(), domain), domain, torch.zeros(VOCAB_SIZE))
    assert report.dense.n == DENSE
    assert report.novel.n == DENSE
    assert report.context.n == len(domain.y)
    assert report.coverage.present == len(set(domain.y[domain.novel(frozenset())].tolist()))
    assert set(report.top1) == set(report.bits) == set(ADJUSTMENTS)
    assert report.top1[0.0] == pytest.approx(1 / report.coverage.present)
    assert report.bits[0.0] == pytest.approx(0.0)


def test_graded() -> None:
    domains = {"a": split(), "b": split(n=8, seed=1)}
    sizes = {name: len(s.novel(frozenset())) for name, s in domains.items()}
    graded = Evaluator(CPU, min_novel=max(sizes.values())).graded(Constant(), domains, torch.zeros(VOCAB_SIZE), 0.0)
    assert set(graded.domains) == {name for name, size in sizes.items() if size == max(sizes.values())}
    hits, labels = next(iter(graded.domains.values()))
    assert hits.shape == labels.shape == (max(sizes.values()),)
    assert torch.equal(hits, labels.eq(0))


def test_point() -> None:
    domains = {"a": split()}
    study = Evaluator(CPU, min_novel=1)
    graded = study.graded(Constant(), domains, torch.zeros(VOCAB_SIZE), 0.5)
    assert study.point(Constant(), domains, torch.zeros(VOCAB_SIZE), 0.5) == study.ruler.point(graded)


def test_grade() -> None:
    evaluator, domains = Evaluator(CPU, min_novel=1), {"a": split()}
    context = {"a": evaluator.logits(Constant(), domains["a"])[Head.CONTEXT]}
    graded = evaluator.grade(context, domains, torch.zeros(VOCAB_SIZE), 0.0)
    expected = evaluator.graded(Constant(), domains, torch.zeros(VOCAB_SIZE), 0.0)
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
    Evaluator(CPU, min_novel=1).report(model, {"a": split()}, torch.zeros(VOCAB_SIZE), 0, 0.0)
    once = model.calls
    Evaluator(CPU).logits(model, split())
    assert model.calls == 2 * once
    domains = {"a": split(), "b": split(n=8, seed=1)}
    report = Evaluator(CPU, min_novel=1).report(Constant(), domains, torch.zeros(VOCAB_SIZE), 3, 0.5)
    assert report.epoch == 3
    assert set(report.domains) == {"a", "b"}
    counted = list(report.domains.values())
    assert report.macro.n == sum(r.novel.n for r in counted)
    mean = np.mean([r.novel.by_tau[0.0] for r in counted])
    assert report.macro.by_tau[ADJUSTMENTS[0]] == pytest.approx(mean)
    assert report.interval.low <= report.macro.by_tau[report.macro.tau] <= report.interval.high
    assert report.top1 == pytest.approx(np.mean([r.top1[report.macro.tau] for r in counted]))
    assert report.bits == pytest.approx(0.0)


def test_summarised() -> None:
    evaluator = Evaluator(CPU, min_novel=1)
    domains = {"a": split(), "b": split(n=8, seed=1)}
    prior = torch.zeros(VOCAB_SIZE)
    logits = {name: evaluator.logits(Constant(), domain) for name, domain in domains.items()}
    summed = evaluator.summarised(logits, domains, prior, 2, 0.25)
    reported = evaluator.report(Constant(), domains, prior, 2, 0.25)
    assert summed.macro == reported.macro
    assert summed.loss == 0.25

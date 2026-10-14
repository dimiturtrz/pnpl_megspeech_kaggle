from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.evaluator import Calibration
from megspeech.evaluation.studies.pool import graph as poolgraph
from megspeech.evaluation.studies.pool.graph import (
    Arm,
    Config,
    Experiment,
    Graph,
    Report,
    Source,
    Study,
    Views,
)
from megspeech.evaluation.studies.pool.knn import SCORED, Dump, Rows
from megspeech.evaluation.studies.pool.layout import Layout
from megspeech.evaluation.studies.pool.smooth import LAMBDAS, TEMPERATURES, Aggregation, Draw, Pool, Smoothing

UNIFORM = -np.log(VOCAB_SIZE)
PAIRS = 16
RULER = Ruler(k=1)
WHOLE = Draw(size=0)
TWIN = Config(Source.EMBEDDING, False, 5, TEMPERATURES[0], Aggregation.LOG, (1.0,))


def rows() -> Rows:
    """Twin rows share a label and an embedding; the first is confidently right, the second leans to a wrong word."""
    labels = torch.arange(PAIRS).repeat_interleave(2)
    base = torch.full((len(labels), VOCAB_SIZE), UNIFORM)
    base[torch.arange(len(labels)), (labels + 1) % VOCAB_SIZE] = 1.0
    base[torch.arange(len(labels)), labels] = torch.tensor([4.0, 0.5]).repeat(PAIRS)
    return Rows(base.log_softmax(-1), labels, (functional.one_hot(labels, PAIRS).float(),))


def smoothing(tau: float = 0.0) -> Smoothing:
    partitions = {p: {"Sherlock1": rows()} for p in SCORED}
    return Smoothing(Dump((), partitions, torch.full((VOCAB_SIZE,), UNIFORM), tau))


def study() -> Study:
    return Study(smoothing(), (smoothing(),))


def layout() -> Layout:
    """Each twin pair is one span."""
    return Layout(torch.arange(2 * PAIRS) // 2, torch.arange(2 * PAIRS) % 2)


def study_with_spans() -> Study:
    return Study(smoothing(), (smoothing(),), {p: {"Sherlock1": layout()} for p in SCORED})


def views() -> Views:
    pool = Pool.drawn(rows(), WHOLE)
    return Views(pool, (pool,), layout().span[None], layout().place[None])


def off_diagonal(sims: torch.Tensor) -> torch.Tensor:
    return sims.masked_fill(torch.eye(sims.shape[-1], dtype=torch.bool), -torch.inf)


def sparse() -> Graph:
    sims = torch.tensor([[[-torch.inf, 0.9, 0.1], [0.9, -torch.inf, 0.8], [0.1, 0.8, -torch.inf]]])
    return Graph.of(sims, 1, 0.1, mutual=False)


def test_cosine() -> None:
    sims = Views.cosine(rows().folds[0][None])
    assert sims[0, 0, 1] == pytest.approx(1.0)
    assert sims[0, 0, 2] < 0.0


def test_zscored() -> None:
    sims = Views.zscored(Views.cosine(rows().folds[0][None]))
    off = ~torch.eye(2 * PAIRS, dtype=torch.bool)
    assert sims[0][off].mean().item() == pytest.approx(0.0, abs=1e-5)
    assert sims[0][off].std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)


def test_embedding() -> None:
    assert views().embedding()[0, 0, 1] == pytest.approx(1.0)


def test_posterior() -> None:
    assert views().posterior()[0, 0, 1] > views().posterior()[0, 0, 2]


def test_two_views() -> None:
    assert off_diagonal(views().two_views()).argmax(-1)[0, 0] == 1


def test_ensemble_embedding() -> None:
    assert torch.allclose(views().ensemble_embedding(), views().embedding(), atol=1e-6)


def test_ensemble_posterior() -> None:
    assert torch.allclose(views().ensemble_posterior(), views().posterior(), atol=1e-6)


def test_ensemble_both() -> None:
    assert off_diagonal(views().ensemble_both()).argmax(-1)[0, 0] == 1


def test_similarity() -> None:
    for source in Source:
        sims = views().similarity(source)
        assert torch.isinf(sims[0].diagonal()).all()
        assert sims.argmax(-1)[0, 0] == 1
        assert views().similarity(source, apart=True)[0, 0, :2].isinf().all()


def test_of() -> None:
    graph = sparse()
    assert graph.keep[0].tolist() == [[False, True, False], [True, False, False], [False, True, False]]
    assert torch.allclose(graph.weights.sum(-1), torch.ones(1, 3))
    mutual = Graph.of(
        torch.tensor([[[-torch.inf, 0.9, 0.1], [0.9, -torch.inf, 0.8], [0.1, 0.8, -torch.inf]]]), 1, 0.1, mutual=True
    )
    assert mutual.keep[0].sum(-1).tolist() == [1, 1, 0]
    assert mutual.weights[0, 2].sum() == 0.0


def test_graph_purity() -> None:
    assert sparse().purity(torch.tensor([[0, 0, 1]])) == pytest.approx((2 / 3, 1.0))


def test_mixed() -> None:
    pool = views().pool
    graph = Graph.of(
        views().embedding().masked_fill(torch.eye(2 * PAIRS, dtype=torch.bool), -torch.inf), 1, 0.1, mutual=False
    )
    mixed = graph.mixed(pool.base, Aggregation.LOG, (1.0,))
    calibration = Calibration(torch.full((VOCAB_SIZE,), UNIFORM), 0.0)
    assert (pool.bacc(mixed, calibration, RULER) - pool.bacc(pool.base, calibration, RULER)).item() == pytest.approx(
        0.5
    )
    alone = Graph(torch.zeros_like(graph.keep), torch.zeros_like(graph.weights))
    assert torch.allclose(alone.mixed(pool.base, Aggregation.PROB, (0.5, 0.5)), pool.base, atol=1e-5)


def test_rounds() -> None:
    assert Config(Source.EMBEDDING, False, 5, 0.1, Aggregation.LOG, (0.1, 0.5)).rounds == 2


def test_admits() -> None:
    arm = Arm("x", (Source.EMBEDDING,), (2,), (True,))
    assert arm.admits(Config(Source.EMBEDDING, True, 5, 0.1, Aggregation.LOG, (0.1, 0.5)))
    assert not arm.admits(TWIN)


def test_configs() -> None:
    single = list(Arm("x", (Source.EMBEDDING,)).configs())
    assert len(single) == len(poolgraph.NEIGHBOURS) * len(TEMPERATURES) * len(Aggregation) * len(LAMBDAS)
    twice = list(Arm("y", (Source.POSTERIOR,), (2,)).configs())
    assert len(twice) == len(poolgraph.PROPAGATED_NEIGHBOURS) * len(TEMPERATURES) * len(Aggregation) * len(LAMBDAS) ** 2


def test_views() -> None:
    drawn = study().views(Partition.TEST, WHOLE)["Sherlock1"]
    assert len(drawn.folds) == 1
    assert drawn.pool.base.shape == (1, 2 * PAIRS, VOCAB_SIZE)


def test_calibration() -> None:
    assert study().calibration().tau == 0.0
    assert study().calibration(0.3).tau == 0.3


def test_grouped() -> None:
    other = Config(Source.EMBEDDING, False, 5, TEMPERATURES[0], Aggregation.PROB, (0.5,))
    grouped = Study.grouped([TWIN, other])
    assert list(grouped[Source.EMBEDDING, False].values()) == [[TWIN, other]]


def test_gains() -> None:
    gains = study().gains(Partition.TEST, WHOLE, [TWIN], RULER)
    assert gains.at(TWIN)["macro"][0] == pytest.approx(0.5, abs=1e-3)


def test_study_purity() -> None:
    purity = study().purity(Partition.TEST, WHOLE, replace(TWIN, mutual=True), k=1)
    assert purity["Sherlock1"] == pytest.approx((1.0, 1.0))
    apart = study_with_spans().purity(Partition.TEST, WHOLE, replace(TWIN, apart=True), k=1)
    assert apart["Sherlock1"][0] == pytest.approx(0.0)


def test_study_confusers() -> None:
    found = study_with_spans().confusers(Partition.TEST, WHOLE, Source.EMBEDDING, k=1)
    assert found.same_span == pytest.approx(1.0)
    assert found.within_1 == pytest.approx(1.0)
    assert found.purity_same_span == pytest.approx(1.0)


def test_retuned() -> None:
    tuned = study().retuned(Partition.TEST, WHOLE, TWIN, [0.0, 0.5], RULER)
    assert set(tuned) == {0.0, 0.5}
    assert tuned[0.0][0] - tuned[0.0][1] == pytest.approx(0.5, abs=1e-3)


def test_taus() -> None:
    assert 0.05 in Study(smoothing(0.05)).taus()


def report() -> Report:
    return Report("arm", TWIN, 0.01, {"macro": (0.02, 0.001)}, {"Sherlock1": (0.4, 0.9)}, {"tau": 0.5})


def test_macro() -> None:
    assert Report.macro({"a": (0.2, 1.0), "b": (0.4, 0.5)}) == pytest.approx((0.3, 0.75))


def test_line() -> None:
    assert "arm" in report().line()
    assert "0.400" in report().line()


def small() -> Arm:
    return Arm("small", (Source.EMBEDDING,))


def test_retune() -> None:
    tuned = Experiment(study(), WHOLE).retune(TWIN)
    assert set(tuned) == {
        "tau",
        "gain at fixed tau",
        "gain at retuned tau, same-tau base",
        "gain at retuned tau, fixed-tau base",
    }


def test_report() -> None:
    experiment = Experiment(study(), WHOLE)
    val = experiment.study.gains(Partition.VALIDATION, WHOLE, list(small().configs()))
    made = experiment.report(small(), val)
    assert made.chosen.source is Source.EMBEDDING
    assert made.purity["Sherlock1"][0] == pytest.approx(1 / poolgraph.PURITY_K)


def test_leaked() -> None:
    assert Arm("x", (Source.ENSEMBLE_POSTERIOR,)).leaked
    assert not small().leaked


def ensemble() -> Arm:
    return Arm("ens", (Source.ENSEMBLE_EMBEDDING,), twin="small")


def test_twinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(poolgraph, "ARMS", (small(), ensemble()))
    experiment = Experiment(study(), WHOLE)
    val = experiment.study.gains(Partition.VALIDATION, WHOLE, [*small().configs(), *ensemble().configs()])
    reports = {a.name: experiment.report(a, val) for a in (small(), ensemble())}
    twinned = experiment.twinned(reports["ens"], reports)
    assert reports["ens"].leaked
    assert twinned.twin_test is not None
    assert experiment.twinned(reports["small"], reports).twin_test is None


def test_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(poolgraph, "ARMS", (small(), ensemble()))
    assert [r.arm for r in Experiment(study(), WHOLE).run()] == ["small", "ens"]


def test_survey() -> None:
    found = study_with_spans().survey()
    assert len(found) == 6
    assert set(found) >= {(Source.EMBEDDING, WHOLE), (Source.POSTERIOR, Draw(grouped=True))}
    whole = study_with_spans().confusers(Partition.TEST, WHOLE, Source.EMBEDDING)
    assert found[Source.EMBEDDING, WHOLE] == whole

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.evaluator import Calibration
from megspeech.evaluation.studies.pool.knn import SCORED, Dump, Mixing, Rows, Space, Study
from megspeech.evaluation.studies.pool.smooth import (
    BASE_SCORES,
    KS,
    LAMBDAS,
    TEMPERATURES,
    TOLERANCE,
    Aggregation,
    Ceiling,
    Collector,
    Draw,
    Gains,
    Near,
    Pool,
    Setting,
    Smoothing,
)

UNIFORM = -np.log(VOCAB_SIZE)
PAIRS = 8
HIDDEN = 4
RULER = Ruler(k=1)


def calibration() -> Calibration:
    return Calibration(torch.full((VOCAB_SIZE,), UNIFORM), 0.0)


def duplicated() -> Pool:
    """Twin rows share a label and an embedding; the first twin is right, the second prefers a wrong word."""
    labels = torch.arange(PAIRS).repeat_interleave(2)
    emb = functional.one_hot(labels, PAIRS).float()
    base = torch.full((len(labels), VOCAB_SIZE), UNIFORM)
    base[torch.arange(len(labels)), (labels + 1) % VOCAB_SIZE] = 1.0
    base[torch.arange(len(labels)), labels] = torch.tensor([4.0, 0.5]).repeat(PAIRS)
    return Pool(base.log_softmax(-1)[None], emb[None], labels[None])


def test_combine() -> None:
    base = torch.tensor([[-1.0, -2.0]])
    log_readout, prob_readout = torch.tensor([[-3.0, -4.0]]), torch.tensor([[0.2, 0.8]])
    assert torch.equal(Aggregation.LOG.combine(base, log_readout, prob_readout, 0.0), base)
    assert torch.allclose(Aggregation.LOG.combine(base, log_readout, prob_readout, 0.5), base + 0.5 * log_readout)
    assert torch.allclose(Aggregation.PROB.combine(base, log_readout, prob_readout, 1.0), prob_readout.log(), atol=1e-5)
    assert torch.allclose(
        Aggregation.PROB.combine(base, log_readout, prob_readout, 0.3), Mixing.LINEAR.mix(base, prob_readout, 0.3)
    )


def test_indices() -> None:
    drawn = Draw(size=5, count=3).indices(20)
    assert drawn.shape == (3, 5)
    assert all(len(set(row.tolist())) == 5 for row in drawn)
    assert torch.equal(drawn, Draw(size=5, count=3).indices(20))
    assert not torch.equal(drawn[0], drawn[1])
    assert Draw(size=0).indices(7).shape == (1, 7)
    assert Draw(size=50, count=3).indices(7).shape == (3, 7)


def test_indices_grouped() -> None:
    span = torch.arange(20) // 4
    drawn = Draw(size=8, count=3, grouped=True).indices(20, span)
    assert drawn.shape == (3, 8)
    assert all(len(set(span[row].tolist())) == 2 for row in drawn)
    assert torch.equal(drawn, Draw(size=8, count=3, grouped=True).indices(20, span))
    assert not torch.equal(drawn[0], drawn[1])
    assert Draw(size=8, count=1, grouped=True).indices(20).shape == (1, 8)


def test_drawn_grouped() -> None:
    rows = Rows(torch.zeros(20, VOCAB_SIZE), torch.arange(20), (torch.ones(20, HIDDEN),))
    span = torch.arange(20) // 4
    pool = Pool.drawn(rows, Draw(size=8, count=2, grouped=True), span)
    assert all(len(set(span[row].tolist())) == 2 for row in pool.y)


def test_weights() -> None:
    near = Near(torch.tensor([[[0.9, 0.1, 0.1]]]), torch.tensor([[[1, 2, 3]]]))
    weights = near.weights(2, 0.1)
    assert weights.shape == (1, 1, 2)
    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[0, 0, 0] > weights[0, 0, 1]


def test_near_purity() -> None:
    y = torch.tensor([[0, 0, 1, 1]])
    near = Near(torch.zeros(1, 4, 1), torch.tensor([[[1], [0], [0], [2]]]))
    assert near.purity(y, 1) == pytest.approx(0.75)


def test_hits() -> None:
    y = torch.tensor([[0, 0, 1, 1]])
    near = Near(torch.zeros(1, 4, 1), torch.tensor([[[1], [0], [0], [2]]]))
    assert near.hits(y, 1).tolist() == [[1.0, 1.0, 0.0, 1.0]]


def test_near_by_label() -> None:
    y = torch.tensor([[0, 0, 1, 1]])
    near = Near(torch.zeros(1, 4, 1), torch.tensor([[[1], [0], [0], [2]]]))
    purity = near.by_label(y, 1)
    assert purity[:2].tolist() == [1.0, 0.5]
    assert purity[2:].isnan().all()


def test_drawn() -> None:
    rows = Rows(torch.zeros(10, VOCAB_SIZE), torch.arange(10), (torch.ones(10, HIDDEN),))
    pool = Pool.drawn(rows, Draw(size=4, count=2))
    assert pool.base.shape == (2, 4, VOCAB_SIZE)
    assert pool.emb.shape == (2, 4, HIDDEN)
    assert torch.allclose(pool.base.exp().sum(-1), torch.ones(2, 4))
    assert pool.y.shape == (2, 4)


def test_near() -> None:
    pool = duplicated()
    near = pool.near(Space.RAW, 3)
    assert near.at.shape == (1, 2 * PAIRS, 3)
    assert not (near.at == torch.arange(2 * PAIRS)[None, :, None]).any()
    assert near.at[0, 0, 0] == 1
    assert near.purity(pool.y, 1) == pytest.approx(1.0)


def test_readouts() -> None:
    pool = duplicated()
    near = pool.near(Space.RAW, 1)
    log_readout, prob_readout = pool.readouts(near, 1, TEMPERATURES[0])
    assert torch.allclose(log_readout[0, 0], pool.base[0, 1])
    assert torch.allclose(prob_readout[0, 0], pool.base[0, 1].exp())


def test_bacc() -> None:
    pool = duplicated()
    perfect = functional.one_hot(pool.y, VOCAB_SIZE).float()
    assert pool.bacc(perfect, calibration(), RULER).item() == pytest.approx(1.0)
    assert pool.bacc(pool.base, calibration(), RULER).item() == pytest.approx(0.5)


def test_settings() -> None:
    pool = duplicated()
    gains = pool.settings(calibration(), RULER, KS[:1])
    assert len(gains) == len(Space) * len(TEMPERATURES) * len(Aggregation) * len(LAMBDAS)
    full = Setting(Space.RAW, KS[0], TEMPERATURES[0], Aggregation.LOG, 1.0)
    assert gains[full].item() == pytest.approx(0.5)
    assert gains[Setting(Space.RAW, KS[0], TEMPERATURES[0], Aggregation.PROB, 1.0)].item() == pytest.approx(0.0)


def gains() -> Gains[Setting]:
    a = Setting(Space.RAW, 5, 0.1, Aggregation.LOG, 0.5)
    b = Setting(Space.RAW, 10, 0.1, Aggregation.LOG, 0.5)
    return Gains(
        {
            "x": {a: torch.tensor([0.1, 0.3]), b: torch.tensor([0.0, 0.0])},
            "y": {a: torch.tensor([0.1, 0.1]), b: torch.tensor([0.0, 0.1])},
        }
    )


def test_gains_macro() -> None:
    macro = gains().macro()
    assert macro[Setting(Space.RAW, 5, 0.1, Aggregation.LOG, 0.5)].tolist() == pytest.approx([0.1, 0.2])


def test_gains_best() -> None:
    assert gains().best().k == 5


def test_at() -> None:
    at = gains().at(Setting(Space.RAW, 5, 0.1, Aggregation.LOG, 0.5))
    assert at["x"][0] == pytest.approx(0.2)
    assert at["x"][1] == pytest.approx(0.1414, abs=1e-3)
    assert at["macro"][0] == pytest.approx(0.15)


def smoothing() -> Smoothing:
    pool = duplicated()
    rows = Rows(pool.base[0], pool.y[0], (pool.emb[0],))
    partitions = {p: {"Sherlock1": rows} for p in SCORED}
    return Smoothing(Dump((), partitions, torch.full((VOCAB_SIZE,), UNIFORM), 0.0))


def test_calibration() -> None:
    assert smoothing().calibration().tau == 0.0


def test_gains() -> None:
    drawn = smoothing().gains(Partition.TEST, Draw(size=0))
    assert set(drawn.domains) == {"Sherlock1"}
    assert drawn.best() in drawn.macro()


def test_smoothing_purity() -> None:
    purity = smoothing().purity(Partition.TEST, Draw(size=0))
    assert purity["Sherlock1"][Space.RAW][KS[0]] <= 1.0


def test_smoothing_by_label() -> None:
    purity = smoothing().by_label(Partition.TEST, Draw(size=0))
    assert len(purity["Sherlock1"][Space.RAW]) == VOCAB_SIZE


def test_chance() -> None:
    assert smoothing().chance(Partition.TEST)["Sherlock1"] == pytest.approx(1 / PAIRS)


def test_posterior() -> None:
    pool = duplicated()
    assert torch.allclose(pool.posterior().emb, pool.base.exp())
    assert torch.equal(pool.posterior().y, pool.y)


def test_same() -> None:
    same = duplicated().same()
    assert same.shape == (1, 2 * PAIRS, 2 * PAIRS)
    assert same.sum(-1).eq(1).all()
    assert same[0, 0, 1] and not same[0, 0, 0]


def test_subset() -> None:
    mask = ~torch.eye(4, dtype=torch.bool)[None]
    chosen = Pool.subset(mask, 2, torch.Generator().manual_seed(0))
    assert chosen.sum(-1).eq(2).all()
    assert not (chosen & ~mask).any()
    assert torch.equal(Pool.subset(mask, 0, torch.Generator().manual_seed(0)), mask)
    assert Pool.subset(mask, 9, torch.Generator().manual_seed(0)).sum(-1).eq(3).all()


def test_chosen() -> None:
    pool = duplicated()
    gain = pool.chosen(pool.same(), calibration(), RULER)
    assert gain[Aggregation.LOG, 1.0].item() == pytest.approx(0.5)
    none = pool.chosen(torch.zeros_like(pool.same()), calibration(), RULER)
    assert all(g.item() == 0.0 for g in none.values())


def test_pools() -> None:
    pools = smoothing().pools(Partition.TEST, Draw(size=0), posterior=True)
    assert torch.allclose(pools["Sherlock1"].emb, pools["Sherlock1"].base.exp())


def test_available() -> None:
    assert smoothing().available(Partition.TEST, Draw(size=0))["Sherlock1"] == pytest.approx(1.0)


def test_ceiling() -> None:
    ceiling = smoothing().ceiling(Partition.TEST, Draw(size=0), (1, 0))
    assert set(ceiling.gains) == {1, 0}
    assert set(ceiling.gains[1]) == {(a, lam) for a in Aggregation for lam in LAMBDAS}


def test_matched() -> None:
    matched = smoothing().matched(Partition.TEST, Draw(size=0), (0.4, 0.6))
    assert set(matched.gains) == {2, 3}


def test_ceiling_macro() -> None:
    mix = (Aggregation.LOG, 1.0)
    macro = Ceiling.macro([{mix: torch.tensor([0.0, 0.2])}, {mix: torch.tensor([0.2, 0.2])}])
    assert macro[mix].tolist() == pytest.approx([0.1, 0.2])


def test_ceiling_best() -> None:
    a, b = (Aggregation.LOG, 0.5), (Aggregation.PROB, 0.5)
    best = Ceiling({3: {a: torch.tensor([0.1, 0.3]), b: torch.tensor([0.0, 0.0])}, 5: {a: torch.tensor([0.2])}}).best()
    assert best[3][0] == a
    assert best[3][1] == pytest.approx(0.2)
    assert best[3][2] == pytest.approx(0.1414, abs=1e-3)
    assert best[5][2] == 0.0


def test_path(tmp_path: Path) -> None:
    assert Collector.path(DataRoot(tmp_path), "base").name == "base.pt"
    assert Collector.path(DataRoot(tmp_path), "base", "fold").name == "base--fold.pt"


def test_reproduces() -> None:
    dump = smoothing().dump
    study = Study(dump)
    own = {p: study.ruler.point(study.base(p)) for p in SCORED}
    assert Collector.reproduces(dump, own) == own
    with pytest.raises(ValueError, match="expected"):
        Collector.reproduces(dump, {p: own[p] + 10 * TOLERANCE for p in SCORED})
    assert set(BASE_SCORES) == set(SCORED)


def test_pick() -> None:
    span = torch.tensor([0, 0, 1, 1, 2, 2])
    rows = Draw(size=4, grouped=True).pick(6, span, 3)
    assert rows.shape == (4,)
    assert torch.equal(rows, Draw(size=4, grouped=True).pick(6, span, 3))
    assert len(set(span[rows].tolist())) == 2
    assert len(set(Draw(size=4).pick(6, None, 3).tolist())) == 4

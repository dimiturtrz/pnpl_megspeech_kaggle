import math
from typing import override

import pytest
import torch
from torch import nn

from megspeech.training.pretrain.objectives import (
    LOSSES,
    Batch,
    Dino,
    IBot,
    Kind,
    KoLeo,
    Loss,
    Objective,
    Regression,
    Setup,
    Term,
)
from megspeech.training.pretrain.prototypes import Assignment, Centring, Prototypes

B, H, T = 6, 8, 20
PROTOTYPES = Prototypes(k=8, bottleneck=6, hidden=10, centre_momentum=0.5)


class Readout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Conv1d(H, H, 1)

    @override
    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.body(maps).transpose(1, 2)


def setup(centring: Centring = Centring.EMA) -> Setup:
    return Setup(Readout(), H, Prototypes(k=8, bottleneck=6, hidden=10, centring=centring), Loss.SMOOTH_L1, 0.05, CPU)


CPU = torch.device("cpu")


def batch(student: torch.Tensor | None = None, progress: float = 0.0) -> Batch:
    torch.manual_seed(0)
    hidden = torch.zeros(B, T, dtype=torch.bool)
    hidden[:, 5:12] = True
    maps = torch.randn(B, H, T) if student is None else student
    return Batch(
        [maps, maps.detach() + 0.1],
        [hidden, hidden.roll(3, 1)],
        torch.randn(B, H, T),
        torch.randn(B, T, H),
        [torch.randn(B, H, T), torch.randn(B, H, T)],
        progress,
    )


def terms(centring: Centring = Centring.EMA) -> list[Term]:
    return [objective.build(setup(centring)) for objective in (Objective(kind) for kind in Kind)]


def test_losses() -> None:
    target = torch.randn(10, 8)
    assert LOSSES[Loss.SMOOTH_L1](target, target) == 0
    assert LOSSES[Loss.COSINE](target, target).abs() < 1e-5
    assert LOSSES[Loss.COSINE](target, -target) > 1.9


def test_parse() -> None:
    assert Objective.parse("ibot") == Objective(Kind.IBOT, 1.0)
    assert Objective.parse("koleo:0.1") == Objective(Kind.KOLEO, 0.1)
    with pytest.raises(ValueError, match="nonsense"):
        Objective.parse("nonsense:1")
    with pytest.raises(ValueError, match="positive"):
        Objective(Kind.DINO, 0.0)


def test_build() -> None:
    built = {objective.kind: objective.build(setup()) for objective in map(Objective, Kind)}
    assert isinstance(built[Kind.REGRESSION], Regression)
    assert isinstance(built[Kind.IBOT], IBot)
    assert isinstance(built[Kind.DINO], Dino)
    assert isinstance(built[Kind.KOLEO], KoLeo)
    assert Objective(Kind.IBOT, 0.3).build(setup()).weight == 0.3
    assert {kind: term.views for kind, term in built.items()} == {
        Kind.REGRESSION: 1,
        Kind.IBOT: 1,
        Kind.DINO: 2,
        Kind.KOLEO: 1,
    }
    assert [term.cross for term in built.values()] == [False, False, True, False]


def test_term_parameters() -> None:
    by_kind = {term.name: term for term in terms()}
    assert by_kind["regression"].parameters() == []
    assert by_kind["koleo"].parameters() == []


def test_distillation_parameters() -> None:
    by_kind = {term.name: term for term in terms()}
    assert all(p.requires_grad for p in by_kind["ibot"].parameters())
    assert by_kind["ibot"].parameters() and by_kind["dino"].parameters()


def test_term_follow() -> None:
    assert Regression("regression", 1.0, setup()).follow(0.9) is None


def test_distillation_follow() -> None:
    term = IBot("ibot", 1.0, setup())
    with torch.no_grad():
        for p in term.head.parameters():
            p.add_(4.0)
    before = [p.clone() for p in term.teacher.parameters()]
    term.follow(0.75)
    moved = zip(term.teacher.parameters(), before, strict=True)
    assert all(torch.allclose(after, was + 1.0) for after, was in moved)


def test_term_floors() -> None:
    assert KoLeo("koleo", 1.0).floors() == {}


def test_regression_floors() -> None:
    assert Regression("regression", 1.0, setup()).floors() == {"spread": 0.05, "target_spread": 0.05}


def test_distillation_floors() -> None:
    assert IBot("ibot", 1.0, setup()).floors() == {"ibot/perplexity": PROTOTYPES.perplexity_floor}
    assert Dino("dino", 1.0, setup()).floors() == {"dino/perplexity": PROTOTYPES.perplexity_floor}


def test_distillation_floor() -> None:
    assert IBot("ibot", 1.0, setup()).floor == PROTOTYPES.perplexity_floor


def test_regression_spread() -> None:
    assert Regression.spread(torch.randn(512, 64))[0] > 0.9
    same = Regression.spread(torch.ones(512, 64) + 1e-4 * torch.randn(512, 64))
    assert same[0] < 0.01
    assert same[1] > 0.99


def test_ko_leo_spread() -> None:
    square = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert torch.allclose(KoLeo.spread(square), torch.full((4,), math.sqrt(2)))
    collinear = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]])
    assert KoLeo.spread(collinear)[0] == pytest.approx(0.0, abs=1e-6)
    assert KoLeo.spread(collinear)[2] == pytest.approx(math.sqrt(2), abs=1e-6)


def test_centred() -> None:
    term = IBot("ibot", 1.0, setup())
    logits = torch.randn(12, 8)
    assert torch.allclose(term.centred(logits, 0.1), (logits / 0.1).softmax(dim=-1))
    term.centre = logits.mean(dim=0)
    assert torch.allclose(term.centred(logits, 0.1), ((logits - logits.mean(dim=0)) / 0.1).softmax(dim=-1))


def test_balanced() -> None:
    term = IBot("ibot", 1.0, setup(Centring.SINKHORN))
    logits = torch.randn(32, 8)
    probs = term.balanced(logits, 0.1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-5)
    assert probs.sum(dim=0).std() < Assignment.sinkhorn(logits, 0.1, 0).sum(dim=0).std()


def test_teach() -> None:
    term = IBot("ibot", 1.0, setup())
    features = torch.randn(30, H)
    keep = torch.arange(30) % 3 == 0
    probs = term.teach(features, 0.0, keep)
    assert probs.shape == (10, 8)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)
    assert not probs.requires_grad
    logits = term.teacher(features)
    assert torch.allclose(term.centre, 0.1 * logits.mean(dim=0), atol=1e-6)
    term.teach(features, 0.0)
    assert torch.allclose(term.centre, 0.9 * 0.1 * logits.mean(dim=0) + 0.1 * logits.mean(dim=0), atol=1e-6)
    assert term.teach(features, 0.0).shape == (30, 8)


def test_readings() -> None:
    term = IBot("ibot", 1.0, setup())
    read = term.readings(torch.full((5, 8), 0.125))
    assert read["ibot/entropy"].item() == pytest.approx(math.log(8), abs=1e-5)
    assert read["ibot/perplexity"].item() == pytest.approx(8.0, rel=1e-4)


def test_distil() -> None:
    term = IBot("ibot", 1.0, setup())
    features = torch.randn(9, H)
    probs = term.teach(features, 0.0)
    keep = torch.arange(9) < 4
    assert term.distil(features, probs).item() > 0
    assert term.distil(features, probs[keep], keep).item() > 0


def trained(kind: Kind, centring: Centring = Centring.EMA) -> list[float]:
    torch.manual_seed(0)
    term = Objective(kind).build(setup(centring))
    reference = batch()
    maps = nn.Parameter(torch.randn(B, H, T))
    params = [maps, *term.parameters()]
    if isinstance(term, Regression):
        params += list(term.predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=0.05)
    seen: list[float] = []
    for _ in range(40):
        current = Batch([maps, maps], reference.hidden, reference.clean, reference.target, reference.across, 0.0)
        loss, readings = term.loss(current)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in readings.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        seen.append(loss.item())
    return seen


def descends(kind: Kind, centring: Centring = Centring.EMA) -> bool:
    seen = trained(kind, centring)
    return sum(seen[-5:]) < sum(seen[:5])


def test_regression_loss() -> None:
    assert descends(Kind.REGRESSION)


def test_i_bot_loss() -> None:
    assert descends(Kind.IBOT)
    assert descends(Kind.IBOT, Centring.SINKHORN)


def test_dino_loss() -> None:
    assert descends(Kind.DINO)
    assert descends(Kind.DINO, Centring.SINKHORN)


def test_loss_readings() -> None:
    by_kind = {term.name: term.loss(batch()) for term in terms()}
    assert set(by_kind["regression"][1]) == {"spread", "target_spread", "target_centre"}
    assert set(by_kind["ibot"][1]) == {"ibot/entropy", "ibot/perplexity"}
    assert set(by_kind["dino"][1]) == {"dino/entropy", "dino/perplexity"}
    assert set(by_kind["koleo"][1]) == {"koleo/nearest"}


def test_ko_leo_loss() -> None:
    maps = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])[:, :, None]
    loss, readings = KoLeo("koleo", 1.0).loss(Batch([maps, maps], [], maps, maps, [], 0.0))
    assert loss.item() == pytest.approx(-math.log(math.sqrt(2)), abs=1e-5)
    assert readings["koleo/nearest"].item() == pytest.approx(math.sqrt(2), abs=1e-5)

import pytest
import torch

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.boosted import LOG_FLOOR
from megspeech.evaluation.studies.crowding import Rows, Section
from megspeech.evaluation.studies.lmrefine import (
    OOV,
    Bank,
    Batch,
    Graded,
    Hidden,
    Oracle,
    Real,
    Simulated,
    Study,
    Trainer,
)
from megspeech.evaluation.studies.refiner import FEATURES, Refiner

ROWS = 64


def states() -> torch.Tensor:
    return torch.tensor([[3, OOV, 4], [5, PAD, PAD]])


def bank() -> Bank:
    labels = torch.cat([torch.arange(VOCAB_SIZE), torch.tensor([IGNORE, IGNORE])])
    beliefs = torch.arange(len(labels)).float()[:, None].expand(-1, VOCAB_SIZE)
    return Bank.of(beliefs, labels)


def real() -> Real:
    labels = torch.where(torch.arange(ROWS) % 5 == 0, IGNORE, torch.arange(ROWS) % VOCAB_SIZE)
    torch.manual_seed(0)
    return Real.of(torch.randn(ROWS, VOCAB_SIZE), labels, torch.arange(ROWS).view(-1, 8)[:, :7])


def test_hidden_drawn() -> None:
    assert not Hidden().drawn(states(), torch.Generator().manual_seed(0)).any()


def test_oracle_drawn() -> None:
    generator = torch.Generator().manual_seed(0)
    shown = Oracle().drawn(states(), generator)
    assert shown.sum(1).tolist() == [2, 0]
    assert not (shown & (states() == PAD)).any()


def test_hidden_views() -> None:
    spans, shown = Hidden().views(states())
    assert spans.tolist() == [0, 1]
    assert not shown.any()


def test_oracle_views() -> None:
    spans, shown = Oracle().views(states())
    assert spans.tolist() == [0, 0, 0, 1]
    assert shown.sum(1).tolist() == [2, 2, 2, 0]
    assert not shown[0, 0]


def batch() -> Batch:
    beliefs = torch.full((2, 3, VOCAB_SIZE), -100.0)
    return Batch(beliefs, states(), torch.tensor([[True, True, False], [False, False, False]]))


def test_inputs() -> None:
    inputs = batch().inputs()
    assert inputs.shape == (2, 3, FEATURES)
    assert inputs[0, 0, 3] == 0.0
    assert inputs[0, 0, 4] == LOG_FLOOR
    assert inputs[0, 1, :VOCAB_SIZE].eq(LOG_FLOOR).all()
    assert inputs[0, 1, VOCAB_SIZE:].tolist() == [1.0, 1.0]
    assert inputs[0, 2, :VOCAB_SIZE].eq(LOG_FLOOR).all()
    assert inputs[0, 2, VOCAB_SIZE:].tolist() == [0.0, 0.0]


def test_targets() -> None:
    assert batch().targets().tolist() == [[IGNORE, IGNORE, 4], [5, IGNORE, IGNORE]]


def test_padding() -> None:
    assert batch().padding().tolist() == [[False, False, False], [False, True, True]]


def test_bank_of() -> None:
    made = bank()
    assert made.counts[OOV] == 2
    assert made.beliefs[made.starts[OOV], 0] == VOCAB_SIZE
    with pytest.raises(ValueError, match="no out-of-fold row"):
        Bank.of(torch.zeros(2, VOCAB_SIZE), torch.tensor([0, 1]))


def test_draw() -> None:
    drawn = bank().draw(states(), torch.Generator().manual_seed(0))
    assert drawn[0, 0, 0] == 3
    assert drawn[0, 1, 0] in (VOCAB_SIZE, VOCAB_SIZE + 1)
    assert drawn[1, 1].eq(0).all()


def simulated() -> Simulated:
    return Simulated(states().to(torch.int8), bank())


def test_simulated_count() -> None:
    assert simulated().count() == 2


def test_real_count() -> None:
    assert real().count() == ROWS // 8


def test_simulated_batch() -> None:
    made = simulated().batch(torch.tensor([1, 0]), Oracle(), torch.Generator().manual_seed(0))
    assert made.states[1].tolist() == [3, OOV, 4]
    assert made.shown[1].sum() == 2


def test_real_batch() -> None:
    rows = real().batch(torch.tensor([0]), Hidden(), torch.Generator())
    assert torch.equal(rows.beliefs[0, 2], real().beliefs[2])
    assert rows.states[0, 0] == OOV


def test_real_of() -> None:
    made = real()
    assert made.states[0] == OOV
    assert torch.allclose(made.beliefs.exp().sum(-1), torch.ones(ROWS))


def test_chains() -> None:
    near = torch.tensor([[PAD, 1], [0, 2], [1, PAD], [PAD, PAD]])
    assert Real.chains(near)[:, :4].tolist() == [[0, 1, 2, PAD], [3, PAD, PAD, PAD]]


def test_take() -> None:
    assert real().take(torch.tensor([True] + [False] * 7)).count() == 1


def test_span_states() -> None:
    assert real().span_states(torch.tensor([0]))[0, :2].tolist() == [OOV, 1]


def test_real_refined() -> None:
    made = real()
    for showing in (Hidden(), Oracle()):
        refined = made.refined(Refiner(), showing)
        assert torch.allclose(refined, made.beliefs, atol=1e-5)


def test_loss() -> None:
    made = real()
    rows = made.index.flatten()
    rows = rows[made.states[rows] < OOV]
    expected = -made.beliefs[rows, made.states[rows]].mean()
    assert made.loss(Refiner(), Hidden()) == pytest.approx(float(expected), abs=1e-5)


def test_step() -> None:
    torch.manual_seed(0)
    model = Refiner()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    made = real().batch(torch.arange(4), Hidden(), torch.Generator())
    first = Trainer(Hidden(), real()).step(model, optimiser, made)
    assert first > Trainer(Hidden(), real()).step(model, optimiser, made) - 1.0
    assert model.out.weight.abs().sum() > 0


def test_train() -> None:
    torch.manual_seed(0)
    trainer = Trainer(Hidden(), real())
    _, best = trainer.train(Refiner(), real(), 1e-3, 4, 2)
    assert len(trainer.curve) == 3
    assert best == min(trainer.curve)


def test_picked() -> None:
    torch.manual_seed(0)
    _, fit = Trainer(Hidden(), real()).picked(Refiner(), real(), 2, 1)
    assert fit["watched"] == min(fit["by_lr"])  # type: ignore[type-var]


def graded() -> Graded:
    made = real()
    ids = torch.arange(0, ROWS, 2)
    labels = torch.where(made.states[ids] == OOV, IGNORE, made.states[ids])
    keep = labels >= 0
    rows = Rows(made.beliefs[ids][keep], made.beliefs[ids][keep], labels[keep])
    return Graded(Section({"a": rows}, {}, torch.zeros(VOCAB_SIZE)), {"a": made}, {"a": ids[keep]})


def test_fused() -> None:
    fused = graded().fused({"a": real().beliefs + 1.0})
    assert torch.allclose(fused["a"].other, torch.ones_like(fused["a"].other))


def test_graded_rebased() -> None:
    made = graded().rebased({"a": torch.zeros(ROWS, VOCAB_SIZE)})
    assert torch.allclose(made.section.domains["a"].head, torch.zeros(1, VOCAB_SIZE).log_softmax(-1))
    assert torch.equal(made.section.domains["a"].labels, graded().section.domains["a"].labels)


def test_graded_refined() -> None:
    refined = graded().refined(Refiner(), Hidden())
    assert torch.allclose(refined["a"].other, torch.zeros_like(refined["a"].other), atol=1e-5)


def test_scored() -> None:
    made = graded()
    result = Study({"base": made, "rebased": made}, made).scored(Refiner(), Hidden(), 0)
    assert set(result) == {"val/base", "val/rebased", "local-test/base"}


def test_held() -> None:
    kept, held = simulated().held(1, torch.Generator().manual_seed(0))
    assert kept.count() == 1
    assert held.count() == 1
    assert held.states.tolist() == [5]
    assert held.index[0].tolist()[:2] == [0, PAD]

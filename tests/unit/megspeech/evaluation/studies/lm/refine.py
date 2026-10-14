from pathlib import Path

import torch

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.evaluation.studies.errors.crowding import Rows, Section
from megspeech.evaluation.studies.lm.refine import Corpus, Graded, Study, Trainer
from megspeech.evaluation.studies.refiner.belief import Refiner
from megspeech.evaluation.studies.refiner.rows import Hidden
from megspeech.evaluation.studies.refiner.wiki import OOV, OUT
from tests.unit.megspeech.evaluation.studies.refiner.rows import ROWS, real


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


def test_wiki(tmp_path: Path) -> None:
    torch.save({"spans": torch.ones(2, 3, dtype=torch.long)}, tmp_path / OUT)
    assert Corpus.wiki(tmp_path, real()).shape == (2, 3)


def test_meg(tmp_path: Path) -> None:
    made = real()
    spans = Corpus.meg(tmp_path, made)
    assert torch.equal(spans, made.span_states(torch.arange(made.count())))


def test_parser() -> None:
    args = Study.parser().parse_args(["m", "--peer", "kf5ac05", "--arms", "meg-only", "--sampler", "shifted"])
    assert (args.peer, args.arms, args.sampler) == ("kf5ac05", ["meg-only"], "shifted")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Placed
from megspeech.evaluation.loaded.ensemble import Scoring
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.studies.ensemble import diversity as module
from megspeech.evaluation.studies.ensemble.diversity import Board, Member, Reads, Relation, Scored, Study
from tests.unit.megspeech.evaluation.loaded.model import CPU, Favourite, loaded, runs

LABELS = torch.arange(8) % VOCAB_SIZE
VAL, TEST = Partition.VALIDATION, Partition.TEST


def reads(favoured: int, strength: float = 5.0) -> Reads:
    """Log-probabilities of 8 words of one domain peaked on class `favoured`."""
    logits = torch.zeros(len(LABELS), VOCAB_SIZE)
    logits[:, favoured] = strength
    return Reads({"d": logits.log_softmax(-1)}, {"d": LABELS})


def board() -> Board:
    return Board(Evaluator(CPU, min_novel=1), torch.zeros(VOCAB_SIZE))


def study(names: list[str], favoured: list[int]) -> Study:
    members = [Member(name, {VAL: reads(f), TEST: reads(f)}) for name, f in zip(names, favoured, strict=True)]
    return Study(members, {VAL: board(), TEST: board()})


def test_of() -> None:
    assert Relation.of("a-s0", "a-s1") is Relation.SEED
    assert Relation.of("ema-a-s0", "ema-b-s0") is Relation.KNOB
    assert Relation.of("ema-a-s0", "acou-a-s0") is Relation.SCHEME


def test_mean() -> None:
    """The mean of equal reads is those reads; of opposed reads, a distribution still normalised."""
    same = Reads.mean([reads(0), reads(0)])
    assert torch.allclose(same.logp["d"], reads(0).logp["d"], atol=1e-6)
    mixed = Reads.mean([reads(0), reads(1)])
    assert torch.allclose(mixed.logp["d"].exp().sum(-1), torch.ones(len(LABELS)))


def test_same_words() -> None:
    other = Reads(reads(0).logp, {"d": LABELS + 1})
    assert reads(0).same_words(reads(1))
    assert not reads(0).same_words(other)


def test_truth() -> None:
    assert reads(0).truth().shape == (len(LABELS),)
    assert reads(0).truth()[0] > reads(0).truth()[1]


def test_score() -> None:
    """A model peaked on class 0 ties every other class, ranked above the truth: only class 0 of 8 hits."""
    scored = board().score(reads(0))
    assert scored.macro == pytest.approx(1 / len(LABELS))
    assert board().score(reads(0), 1.0).tau == 1.0


def test_collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    every = runs(tmp_path)
    monkeypatch.setattr(module.Loader, "of", staticmethod(lambda *_: replace(loaded(tmp_path), model=Favourite())))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    member, boards = Member.collect(Placed(loaded(tmp_path).root, "x", CPU), Scoring())
    assert member.name == "x"
    assert set(member.reads) == set(boards) == {VAL, TEST}
    assert isinstance(Run, type)


def test_ensemble() -> None:
    one = study(["a", "b"], [0, 1])
    assert one.ensemble(["a"], VAL).logp["d"].argmax(-1).tolist() == [0] * len(LABELS)


def test_single() -> None:
    val, test = study(["a"], [0]).single(study(["a"], [0]).members[0])
    assert val == Scored(1 / len(LABELS), val.tau)
    assert test.tau == val.tau


def test_pair_scores() -> None:
    matrix = study(["a", "b"], [0, 1]).pair_scores()
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == matrix[1, 0]


def test_correlation() -> None:
    matrix = study(["a", "b", "c"], [0, 0, 1]).correlation(VAL)
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[0, 2] < 1.0


def test_greedy() -> None:
    steps = study(["a-s0", "b-s0", "c-s0"], [0, 1, 2]).greedy(limit=2)
    assert [len(step.names) for step in steps] == [1, 2]
    assert len(set(steps[-1].names)) == 2


def test_groups() -> None:
    grouped = study(["ema-a-s0", "ema-a-s1", "acou-w-s0"], [0, 0, 1]).groups(np.eye(3))
    assert {relation: len(values) for relation, values in grouped.items()} == {Relation.SEED: 1, Relation.SCHEME: 2}


def test_summary() -> None:
    text = study(["a-s0", "a-s1"], [0, 0]).summary(np.full((2, 2), 0.5))
    assert "seed-seed n=1 mean 0.5000" in text


def test_table() -> None:
    assert study(["a", "b"], [0, 1]).table(np.eye(2)) == [" 0 1.000 0.000", " 1 0.000 1.000"]


def test_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger=module.LOG.name)
    study(["a-s0", "a-s1"], [0, 1]).log()
    text = caplog.text
    assert "single  0 a-s0" in text
    assert "greedy 2" in text

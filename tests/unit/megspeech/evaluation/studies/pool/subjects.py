import logging
import math
import sys
from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.pool import subjects as module
from megspeech.evaluation.studies.pool.subjects import CHANCE, AlignedSubjects, Equal, Pool, Reliable, main
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded.model import NAMES, POOLED, SPOKEN, TRAINED, loaded, runs


def test_equal_weights() -> None:
    assert Equal().weights(torch.tensor([0.1, 0.9])).tolist() == [1.0, 1.0]


def test_reliable_weights() -> None:
    weights = Reliable().weights(torch.tensor([CHANCE - 0.1, CHANCE + 0.1, CHANCE + 0.3]))
    assert weights.tolist() == pytest.approx([0.0, 0.75, 2.25])
    assert Reliable().weights(torch.full((2,), CHANCE)).tolist() == [0.0, 0.0]


def test_combine() -> None:
    generator = torch.Generator().manual_seed(0)
    reference, others = (
        torch.randn(3, VOCAB_SIZE, generator=generator),
        torch.randn(2, 3, VOCAB_SIZE, generator=generator),
    )
    aligned = AlignedSubjects(others, torch.tensor([[True, True, False], [True, False, False]]), torch.ones(2))
    assert torch.allclose(Pool(math.inf, Equal()).combine(reference, aligned), reference.log_softmax(-1))
    pooled = Pool(2.0, Equal()).combine(reference, aligned)
    logs = reference.log_softmax(-1), others.log_softmax(-1)
    assert torch.allclose(pooled[0], (2 * logs[0][0] + logs[1][0, 0] + logs[1][1, 0]) / 4)
    assert torch.allclose(pooled[1], (2 * logs[0][1] + logs[1][0, 1]) / 3)
    assert torch.allclose(pooled[2], logs[0][2])


def test_heard() -> None:
    present = torch.tensor([[True, False, False], [True, True, False]])
    assert AlignedSubjects(torch.zeros(2, 3, VOCAB_SIZE), present, torch.ones(2)).heard.tolist() == [True, True, False]


def test_aligned(tmp_path: Path) -> None:
    every = runs(tmp_path)
    _, reference = loaded(tmp_path).domains({RULER: every[:1]})
    logits, present, reliability = AlignedSubjects.aligned(
        loaded(tmp_path), reference.iloc[:4], every[POOLED], every[TRAINED]
    )
    assert present.tolist() == [True] * 4
    assert logits.argmax(dim=1).tolist() == [VOCAB_SIZE - 1] * 4
    assert reliability == pytest.approx(0.0)


def test_of(tmp_path: Path) -> None:
    every = runs(tmp_path)
    _, reference = loaded(tmp_path).domains({RULER: every[:1]})
    aligned = AlignedSubjects.of(loaded(tmp_path), reference, every)
    assert aligned.logits.shape == (1, len(SPOKEN), VOCAB_SIZE)
    assert aligned.present.all()
    assert aligned.reliability.shape == (1,)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    every = runs(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(module.Loader, "of", staticmethod(lambda *_: loaded(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["pooling", NAMES[0]])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        main()
    assert f"ruler words {len(SPOKEN)}, heard by another subject {len(SPOKEN)}" in caplog.text
    assert caplog.text.count("macro") == 2 * len(module.WEIGHTS)

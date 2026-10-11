import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.corpus import Corpus
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.training.cli import throughput as module
from megspeech.training.cli.throughput import EpochEstimate, Phase, Throughput, main
from megspeech.training.config import RunConfig, SelectionConfig
from megspeech.training.loop.trainer import Trainer
from megspeech.training.optim import OptimConfig

CPU = torch.device("cpu")
SAMPLES = 50


def corpus(n: int = 16) -> Corpus:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    split = Split.from_arrays(x, np.arange(n) % 3, Spans.by_gap(np.arange(n) * 0.2 + np.arange(n) // 8 * 5.0), CPU)
    return Corpus(split, {"a": split}, frozenset())


def projection() -> EpochEstimate:
    return EpochEstimate(Phase(2.0, 1.0, 0.0, ""), Phase(10.0, 1.0, 0.0, ""), 0.1, 100)


def test_epoch() -> None:
    assert projection().epoch() == pytest.approx(20.0)


def test_eval_share() -> None:
    assert projection().eval_share() == pytest.approx(0.5)


def test_phase() -> None:
    phase = Throughput(rows=3, host=True).phase(lambda: torch.ones(64, 64) @ torch.ones(64, 64), CPU)
    assert phase.seconds > 0
    assert phase.device == 0
    assert "mm" in phase.table


def test_measure() -> None:
    out = Throughput(steps=2, warmup=1).measure(
        Trainer(RunConfig(optim=OptimConfig(batch=4)), Evaluator(CPU, min_novel=1)), corpus(), 40
    )
    assert out.steps == 10
    assert out.per_step == pytest.approx(out.train.seconds / 2)
    assert 0 < out.eval_share() < 1


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(module.Corpus, "load", classmethod(lambda *_: corpus(SelectionConfig.min_novel)))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: tmp_path))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["throughput", "--steps", "1", "--full-spans", "8"])
    with caplog.at_level("INFO"):
        main()
    assert "eval share" in caplog.text

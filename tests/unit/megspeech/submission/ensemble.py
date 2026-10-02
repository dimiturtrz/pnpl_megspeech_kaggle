import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.corpus import METRICS, DataConfig
from megspeech.data.runs import CHANNELS, Run
from megspeech.data.spans import Spans
from megspeech.data.split import DENSE, Split
from megspeech.data.vocabulary import WORDS
from megspeech.evaluation.report import Evaluator
from megspeech.submission import ensemble as module
from megspeech.submission.ensemble import main

CPU = torch.device("cpu")
SAMPLES = 50
ROWS = 12


class Constant(nn.Module):
    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(*slot.shape, WORDS)
        logits[..., 0] = 1
        return logits, logits


def validation() -> dict[str, Split]:
    x = np.random.default_rng(0).normal(size=(ROWS, CHANNELS, SAMPLES)).astype(np.float32)
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(ROWS - DENSE) * 2.0]
    return {"Sherlock1": Split.from_arrays(x, np.arange(ROWS) % 3, Spans.by_gap(onset), CPU)}


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[tuple[int, Path]] = []
    train = Run(tmp_path / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5")
    devices = SimpleNamespace(device=lambda _: CPU, tensor=torch.tensor, ones=torch.ones)
    monkeypatch.setattr(module, "torch", devices)
    monkeypatch.setattr(module, "Evaluator", functools.partial(Evaluator, min_novel=1))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: [train]))
    heard = pd.DataFrame({"run": "r", "sentence": 0, "onset": [0.0, 0.1], "target": [0, 1], "token": [7, 8]})
    monkeypatch.setattr(module.Corpus, "heard", classmethod(lambda *_: heard))
    monkeypatch.setattr(module.Continuous, "load", classmethod(lambda *_, **__: SimpleNamespace(words=heard)))
    monkeypatch.setattr(module.Corpus, "words", staticmethod(lambda table, *_: table))
    monkeypatch.setattr(module.Corpus, "domains", classmethod(lambda *_: validation()))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: Constant()))
    monkeypatch.setattr(module.Submitter, "submit", lambda self, *args: written.append((self.data, args[-1])))
    for name in ("a", "b"):
        member = DataRoot(tmp_path).models() / name
        member.mkdir(parents=True)
        (member / METRICS).write_text(json.dumps({"config": {"data": {"window": 2.0, "standardised": True}}}))
    monkeypatch.setattr(sys, "argv", ["ensemble", "--out", "both", "a", "b"])
    main()
    out = DataRoot(tmp_path).models() / "both"
    metrics = json.loads((out / METRICS).read_text())
    assert metrics["members"] == ["a", "b"]
    assert metrics["config"]["data"]["window"] == 2.0
    assert set(metrics["best"]["domains"]) == {"Sherlock1"}
    assert written == [(DataConfig(window=2.0, standardised=True), out / "submission.csv")]

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
from megspeech.data.batching.split import DENSE, Split
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.pooling import Pooling, Scoring
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
        logits = torch.zeros(*slot.shape, VOCAB_SIZE)
        logits[..., 0] = 1
        return logits, logits


def validation() -> dict[str, Split]:
    x = np.random.default_rng(0).normal(size=(ROWS, CHANNELS, SAMPLES)).astype(np.float32)
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(ROWS - DENSE) * 2.0]
    return {"Sherlock1": Split.from_arrays(x, np.arange(ROWS) % 3, Spans.by_gap(onset), CPU)}


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[tuple[int, Path]] = []
    loaded: list[tuple[list[Path], Scoring]] = []
    data = DataConfig(window=2.0, standardised=True)

    def pooled(_: type[Pooling], root: DataRoot, directories: list[Path], scoring: Scoring) -> Pooling:
        loaded.append((directories, scoring))
        return Pooling(root, Constant(), data, Evaluator(CPU, min_novel=1), torch.zeros(VOCAB_SIZE))

    monkeypatch.setattr(module, "torch", SimpleNamespace(device=lambda _: CPU))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: []))
    monkeypatch.setattr(module.Pooling, "loaded", classmethod(pooled))
    monkeypatch.setattr(module.Pooling, "domains", lambda *_: (validation(), pd.DataFrame()))
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
    assert written == [(data, out / "submission.csv")]
    assert loaded == [([DataRoot(tmp_path).models() / "a", DataRoot(tmp_path).models() / "b"], loaded[0][1])]
    assert (loaded[0][1].device, loaded[0][1].shifts) == (CPU, (0,))

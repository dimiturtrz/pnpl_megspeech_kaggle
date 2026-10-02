import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.corpus import Corpus
from megspeech.data.runs import CHANNELS
from megspeech.data.spans import Spans
from megspeech.data.split import Split
from megspeech.training import train as module
from megspeech.training.config import METRICS
from megspeech.training.train import main

CPU = torch.device("cpu")
SAMPLES = 50
ROWS = 16


def corpus() -> Corpus:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(ROWS, CHANNELS, SAMPLES)).astype(np.float32)
    onset = np.arange(ROWS) * 0.2 + np.arange(ROWS) // 8 * 5.0
    split = Split.from_arrays(x, np.arange(ROWS) % 3, Spans.by_gap(onset), CPU)
    return Corpus(split, {"a": split}, frozenset())


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[Path] = []
    monkeypatch.setattr(module.Corpus, "load", classmethod(lambda *_: corpus()))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Submitter, "submit", lambda _self, *args: written.append(args[-1]))
    argv = ["train", "--name", "run", "--optim.epochs", "1", "--optim.batch", "4", "--selection.min_novel", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    out = DataRoot(tmp_path).models() / "run"
    metrics = json.loads((out / METRICS).read_text())
    assert metrics["train"] == ROWS
    assert len(metrics["history"]) == 1
    assert (out / "model.pt").is_file()
    assert (out / "hits.pt").is_file()
    assert written == [out / "submission.csv"]

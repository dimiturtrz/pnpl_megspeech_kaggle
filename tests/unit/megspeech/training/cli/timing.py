import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.corpus import Corpus
from megspeech.data.spans.spans import Spans
from megspeech.model.context.timing_only import FILE
from megspeech.training.cli import timing as module
from megspeech.training.cli.timing import main
from megspeech.training.config import METRICS, RunConfig
from megspeech.training.loop import trainer as trainer_module

CPU = torch.device("cpu")
ROWS = 16


def corpus() -> Corpus:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(ROWS, CHANNELS, 50)).astype(np.float32)
    onset = np.arange(ROWS) * 0.2 + np.arange(ROWS) // 8 * 5.0
    split = Split.from_arrays(x, np.arange(ROWS) % 3, Spans.by_gap(onset), CPU)
    positions = pd.DataFrame({"run": "a", "position": np.arange(ROWS), "onset": np.arange(ROWS, dtype=float)})
    return Corpus(split, {"a": split}, frozenset(), positions)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = DataRoot(tmp_path)
    (root.models() / "origin").mkdir(parents=True)
    (root.models() / "origin" / METRICS).write_text(json.dumps({"config": asdict(RunConfig())}))
    monkeypatch.setattr(trainer_module.Corpus, "load", classmethod(lambda *_: corpus()))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: root))
    argv = ["timing", "--origin", "origin", "--name", "t", "--epochs", "1", "--batch", "4"]
    monkeypatch.setattr(sys, "argv", [*argv, "--device", "cpu", "--share", "0.5"])
    main()
    assert (root.models() / "t" / FILE).is_file()
    summary = json.loads((root.models() / "t" / METRICS).read_text())
    assert summary["expert"]["epochs"] == 1
    assert summary["config"]["data"]["train_share"] == 0.5

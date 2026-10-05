import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.corpus import Corpus
from megspeech.data.labels.context_tags import ContextTags
from megspeech.data.spans.spans import Spans
from megspeech.training.cli import train as module
from megspeech.training.cli.train import HEADS, main
from megspeech.training.config import METRICS

CPU = torch.device("cpu")
SAMPLES = 50
ROWS = 16


def corpus() -> Corpus:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(ROWS, CHANNELS, SAMPLES)).astype(np.float32)
    onset = np.arange(ROWS) * 0.2 + np.arange(ROWS) // 8 * 5.0
    split = Split.from_arrays(x, np.arange(ROWS) % 3, Spans.by_gap(onset), CPU)
    positions = pd.DataFrame({"run": "a", "position": np.arange(ROWS), "onset": np.arange(ROWS, dtype=float)})
    return Corpus(split, {"a": split}, frozenset(), positions)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[Path] = []
    monkeypatch.setattr(module.Corpus, "load", classmethod(lambda *_: corpus()))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.KaggleWriter, "write", lambda _self, *args: written.append(args[-1]))
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
    assert not (out / HEADS).exists()
    ContextTags(np.array(["a"] * ROWS), np.arange(ROWS, dtype=float), np.arange(ROWS) % 12).save(
        ContextTags.path(DataRoot(tmp_path))
    )
    tagged = ["--context.conv.head", "tagged", "--optim.tagging", "1.0"]
    monkeypatch.setattr(sys, "argv", [*argv[:2], "tagged", *argv[3:], *tagged])
    main()
    assert any(key.startswith("heads.") for key in torch.load(DataRoot(tmp_path).models() / "tagged" / HEADS))
    assert any(
        key.startswith("words.head.tag.") for key in torch.load(DataRoot(tmp_path).models() / "tagged" / "model.pt")
    )
    narrow = ["--context.width", "32", "--context.conv.hidden", "32"]
    monkeypatch.setattr(sys, "argv", [*argv[:2], "base", *argv[3:], *narrow])
    main()
    monkeypatch.setattr(sys, "argv", [*argv[:2], "tuned", *argv[3:], "--optim.warm", "base"])
    main()
    tuned = json.loads((DataRoot(tmp_path).models() / "tuned" / METRICS).read_text())["config"]
    assert tuned["context"]["width"] == 32
    assert tuned["optim"]["warm"] == "base"
    monkeypatch.setattr(sys, "argv", ["train", "--help"])
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: pytest.fail("help read paths.yaml")))
    with pytest.raises(SystemExit):
        main()

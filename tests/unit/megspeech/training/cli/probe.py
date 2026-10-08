import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.paths import DataRoot
from megspeech.training.cli import probe as module
from megspeech.training.cli.probe import main

SCORE = 0.4321


class Stub:
    def to(self, _device: torch.device) -> "Stub":
        return self

    def score(self, _net: "Stub") -> float:
        return SCORE


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    data = SimpleNamespace(
        run_cache=lambda _root: tmp_path,
        standardised=True,
        geometry=lambda _root, _device: None,
    )
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.RunConfig, "saved", classmethod(lambda _c, _directory: SimpleNamespace(data=data)))
    monkeypatch.setattr(module.Run, "everywhere", classmethod(lambda _c, _root: []))
    monkeypatch.setattr(module.Sessions, "of", classmethod(lambda _c, _data, _runs: SimpleNamespace(names=[])))
    monkeypatch.setattr(module.DeviceRecording, "load", classmethod(lambda _c, *_a, **_k: None))
    monkeypatch.setattr(module.Checkpoint, "load", classmethod(lambda _c, _path: Stub()))
    monkeypatch.setattr(module.Progress, "load", classmethod(lambda _c, *_a: Stub()))
    monkeypatch.setattr(sys, "argv", ["probe", "--model", "run", "--words", "7", "--k", "3"])
    with caplog.at_level(logging.INFO):
        main()
    assert f"probe run BAcc@10 {SCORE:.4f} (chance 0.2, words 7, k 3)" in caplog.text

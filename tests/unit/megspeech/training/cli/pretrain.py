import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.constants import CHANNELS
from megspeech.training.cli import pretrain as module
from megspeech.training.cli.pretrain import main
from megspeech.training.config import METRICS
from megspeech.training.pretrain.pretrain import Pretrainer
from megspeech.training.pretrain.windows import Pool


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = DeviceRecording(torch.randn(CHANNELS, 400).half(), np.array([0, 400]), ("a",), pd.DataFrame())
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(
        module.Pretrainer,
        "of",
        classmethod(lambda _c, config, _root, device: Pretrainer(config, device, Pool(recording, 50))),
    )
    argv = ["pretrain", "--name", "ssl", "--epochs", "1", "--batch", "4", "--windows", "8", "--fit", "8"]
    narrow = ["--collapse_floor", "0", "--top", "1", "--context.width", "16", "--context.heads", "2"]
    monkeypatch.setattr(sys, "argv", [*argv, *narrow, "--context.conv.hidden", "8"])
    main()
    out = DataRoot(tmp_path).models() / "ssl"
    assert (out / "model.pt").is_file()
    assert len(json.loads((out / METRICS).read_text())["history"]) == 1

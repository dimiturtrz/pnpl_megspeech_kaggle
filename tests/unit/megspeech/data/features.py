from pathlib import Path

import numpy as np
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.features import FEATURES, TRAIN, Features
from megspeech.data.spans import Spans
from megspeech.data.split import Split
from megspeech.data.windows import Windows

FEATURE = torch.arange(6.0).reshape(3, 2)


def saved(tmp_path: Path) -> Features:
    features = Features(tmp_path / "f")
    features.save(TRAIN, FEATURE)
    return features


def windows(count: int = 3) -> Windows:
    return Windows.stacked(torch.ones(count, 4, 5))


def test_under(tmp_path: Path):
    root = DataRoot(tmp_path)
    assert Features.under(root, "x").directory == root.kind("models") / FEATURES / "x"
    assert Features.under(root, "").directory is None


def test_path(tmp_path: Path):
    assert Features(tmp_path).path(TRAIN) == tmp_path / "train.pt"
    with pytest.raises(ValueError, match="no features"):
        Features().path(TRAIN)


def test_save(tmp_path: Path):
    path = saved(tmp_path).path(TRAIN)
    assert torch.equal(torch.load(path, weights_only=True), FEATURE)


def test_read(tmp_path: Path):
    raw = windows()
    assert Features().read(TRAIN, raw) is raw
    read = saved(tmp_path).read(TRAIN, raw)
    assert torch.equal(read(torch.tensor([2, 0])), FEATURE[[2, 0], :, None])
    resolved = Features(tmp_path / "r")
    resolved.save(TRAIN, FEATURE.reshape(3, 1, 2))
    assert torch.equal(resolved.read(TRAIN, raw)(torch.tensor([1])), FEATURE[[1], None])
    with pytest.raises(ValueError, match="2 windows"):
        saved(tmp_path).read(TRAIN, windows(2))


def test_split(tmp_path: Path):
    split = Split.of(windows(), (np.zeros(3, np.int64),) * 2, Spans.by_gap(np.array([0.0, 0.1, 9.0])), np.ones(3))
    read = saved(tmp_path).split(TRAIN, split)
    assert read.windows.length == 1
    assert read.y is split.y

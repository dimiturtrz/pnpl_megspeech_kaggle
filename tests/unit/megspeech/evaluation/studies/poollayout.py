from pathlib import Path

import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.studies.poollayout import Layout

CPU = torch.device("cpu")


def split() -> Split:
    onset = np.array([0.0, 0.2, 0.4, 10.0, 10.2, 20.0])
    x = np.zeros((6, 4, 4), dtype=np.float32)
    return Split.from_arrays(x, np.arange(6), Spans.by_gap(onset), CPU)


def test_of() -> None:
    layout = Layout.of(split(), torch.tensor([0, 1, 3, 4, 5]))
    assert layout.span.tolist() == [0, 0, 1, 1, 2]
    assert layout.place.tolist() == [0, 1, 0, 1, 0]


def test_alone() -> None:
    layout = Layout.alone(4)
    assert layout.span.tolist() == [0, 1, 2, 3]
    assert layout.place.tolist() == [0, 0, 0, 0]


def test_path(tmp_path: Path) -> None:
    assert Layout.path(DataRoot(tmp_path), "base").name == "base--layout.pt"

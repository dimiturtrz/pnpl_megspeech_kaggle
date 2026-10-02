from pathlib import Path

import torch

from megspeech.onsets import CONTEXT as DEFAULT
from megspeech.onsets import ContextConfig, OnsetEncoding

CONTEXT = ContextConfig(width=16, timescales=4)


def test_onset_encoding_forward() -> None:
    out = OnsetEncoding(CONTEXT).forward(torch.tensor([[0.0, 0.3], [0.0, 0.0]]))
    assert out.shape == (2, 2, CONTEXT.width)
    assert torch.equal(out[0, 0], out[1, 1])


def test_save(tmp_path: Path) -> None:
    assert CONTEXT.save(tmp_path).exists()


def test_load(tmp_path: Path) -> None:
    assert ContextConfig.load(tmp_path) == DEFAULT
    CONTEXT.save(tmp_path)
    assert ContextConfig.load(tmp_path) == CONTEXT

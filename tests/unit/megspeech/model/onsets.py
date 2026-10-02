from pathlib import Path

import torch

from megspeech.model.onsets import CONTEXT as DEFAULT
from megspeech.model.onsets import HOLDOUT_GAP, Clock, ContextConfig, OnsetEncoding
from megspeech.objective.heads import Objective

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
    ordered = ContextConfig(clock=Clock.ORDER)
    ordered.save(tmp_path)
    assert ContextConfig.load(tmp_path).clock is Clock.ORDER
    retrieval = ContextConfig(objective=Objective.SIGLIP)
    retrieval.save(tmp_path)
    assert ContextConfig.load(tmp_path).objective is Objective.SIGLIP


def test_times() -> None:
    onset = torch.tensor([[0.0, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(Clock.ONSET.times(onset, mask), onset)
    assert torch.allclose(Clock.ORDER.times(onset, mask)[0, :2], torch.tensor([0.0, HOLDOUT_GAP]))

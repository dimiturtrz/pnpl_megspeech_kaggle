from pathlib import Path

import torch

from core.paths import DataRoot
from megspeech.model.encoder.motion import Frozen, Moving
from megspeech.training.transfer import WEIGHTS, TransferConfig

CPU = torch.device("cpu")


def root(tmp_path: Path) -> DataRoot:
    made = DataRoot(tmp_path)
    (made.models() / "pre").mkdir(parents=True)
    torch.save({"w": torch.ones(2)}, made.models() / "pre" / WEIGHTS)
    return made


def test_weights(tmp_path: Path) -> None:
    saved = root(tmp_path)
    warm, stem = TransferConfig(warm="pre").weights(saved, CPU)
    assert torch.equal(warm["w"], torch.ones(2))
    assert stem == {}
    warm, stem = TransferConfig(stem="pre").weights(saved, CPU)
    assert (warm, set(stem)) == ({}, {"w"})


def test_weights_of_nothing_never_read_the_root() -> None:
    assert TransferConfig().weights(DataRoot(Path("absent")), CPU) == ({}, {})


def test_motion() -> None:
    assert type(TransferConfig(encoder_lr=0.0).motion()) is Frozen
    assert type(TransferConfig(encoder_lr=0.1).motion()) is Moving

import sys
from pathlib import Path

import pytest

from core.paths import DataRoot
from megspeech.training.cli import arm as module
from megspeech.training.cli.arm import TRAIN, WEIGHTS, Arm

ARM = Arm("base", 2, "h512", ("--context.conv.hidden", "512"))


def test_stem() -> None:
    assert ARM.stem == "base-h512-s2"


def test_fine() -> None:
    assert ARM.fine == "base-h512-s2-ft-s2"


def test_commands(tmp_path: Path) -> None:
    stem, fine = ARM.commands(tmp_path)
    shared = ["--optim.seed", "2", "--context.conv.hidden", "512"]
    assert stem == [*TRAIN, "--origin", "base-s2", *shared, "--name", "base-h512-s2"]
    assert fine == [*TRAIN, "--origin", "base-s2-ft-s2", "--optim.stem", "base-h512-s2", *shared, "--name", ARM.fine]
    (tmp_path / ARM.stem).mkdir()
    assert len(ARM.commands(tmp_path)) == len([stem, fine])
    (tmp_path / ARM.stem / WEIGHTS).write_bytes(b"")
    assert ARM.commands(tmp_path) == [fine]


def test_commands_stem_flags(tmp_path: Path) -> None:
    short = Arm("base", 2, "stem20", (), ("--optim.epochs", "20"))
    stem, fine = short.commands(tmp_path)
    assert stem[-4:-2] == ["--optim.epochs", "20"]
    assert "--optim.epochs" not in fine


def test_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, *, check: ran.append(command) if check else None)
    ARM.run(tmp_path)
    assert ran == ARM.commands(tmp_path)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[tuple[Arm, Path]] = []
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(Arm, "run", lambda arm, models: runs.append((arm, models)))
    argv = [
        "arm",
        "--base",
        "base",
        "--seed",
        "2",
        "--tag",
        "h512",
        "--stem-flags=",
        "--",
        "--context.conv.hidden",
        "512",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    Arm.main()
    assert runs == [(ARM, DataRoot(tmp_path).models())]

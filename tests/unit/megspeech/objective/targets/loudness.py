from pathlib import Path
from typing import cast

import numpy as np
import pytest

from core.paths import DataRoot
from megspeech.data.config import Cut
from megspeech.data.raw.runs import Run
from megspeech.objective.targets import loudness as module
from megspeech.objective.targets.loudness import CUT, LENGTH, NAME, OFFSET, Loudness
from megspeech.objective.targets.speech import Heard
from tests.unit.megspeech.data.config import HELD, TRAIN
from tests.unit.megspeech.objective.targets.speech import SECONDS, Root, heard_run, loud


def test_of() -> None:
    """Frames of silence then tone: standardised over the chapter, the tone's frames above the silence's."""
    levels = Loudness.of(loud(SECONDS, 1.0)).levels
    assert len(levels) == SECONDS * 50
    assert levels.mean() == pytest.approx(0.0, abs=1e-4)
    assert levels.std() == pytest.approx(1.0, abs=1e-4)
    assert levels[60:].min() > levels[:40].max()


def test_windows() -> None:
    envelope = Loudness(np.arange(100, dtype=np.float32))
    targets = envelope.windows(np.array([0.0, 0.5, 1.02, np.nan, -0.1]), CUT)
    assert targets.present.tolist() == [True, True, False, False, False]
    assert targets.vectors[1].tolist() == (np.arange(LENGTH) + 25 + OFFSET).tolist()
    wide = envelope.windows(np.array([0.0, 0.5, 0.52]), Cut(75, 0))
    assert wide.present.tolist() == [True, True, False]
    assert wide.vectors[0].tolist() == np.arange(75).tolist()


def test_directory(tmp_path: Path) -> None:
    root = cast("DataRoot", Root(tmp_path))
    assert Loudness.directory(root, CUT) == tmp_path / "processed" / "speech" / NAME
    assert Loudness.directory(root, Cut(75, 0)) == tmp_path / "processed" / "speech" / f"{NAME}-75-0"


def test_parsed() -> None:
    assert Loudness.parsed([]) == CUT
    assert Loudness.parsed(["--window", "1.5"]) == (75, 0)
    assert Loudness.parsed(["--window", "1.5", "--delay", "0.2", "other"]) == (75, 10)


def test_targets(tmp_path: Path) -> None:
    """The first word starts the tone, so its window is the tone's; the second's runs past the chapter's end."""
    targets = Loudness.targets(Heard(heard_run(tmp_path)), CUT)
    assert targets.vectors.shape == (2, LENGTH)
    assert targets.present.tolist() == [True, False]
    levels = Loudness.of(loud(SECONDS, 1.0)).levels
    assert np.allclose(targets.vectors[0].astype(np.float32), levels[50 : 50 + LENGTH], atol=1e-2)


def test_write(tmp_path: Path) -> None:
    heard_run(tmp_path)
    heard_run(tmp_path, HELD)
    root = cast("DataRoot", Root(tmp_path))
    directory = Loudness.write(root, CUT)
    assert [path.stem for path in directory.glob("*.npy")] == [Run(Path(TRAIN)).name]
    assert Loudness.write(root, CUT) == directory


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard_run(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    monkeypatch.setattr("sys.argv", ["loudness"])
    Loudness.main()
    assert np.load(tmp_path / "processed" / "speech" / NAME / f"{Run(Path(TRAIN)).name}.npy").shape == (2, LENGTH)
    monkeypatch.setattr("sys.argv", ["loudness", "--window", "1.5"])
    Loudness.main()
    assert (tmp_path / "processed" / "speech" / f"{NAME}-75-0" / f"{Run(Path(TRAIN)).name}.npy").exists()

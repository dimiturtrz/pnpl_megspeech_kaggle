from pathlib import Path
from typing import cast

import numpy as np
import pytest

from core.paths import DataRoot
from megspeech.data.raw.runs import NO_SENTENCE, Run
from megspeech.objective.targets import sentence as module
from megspeech.objective.targets.sentence import SUFFIX, Sentences
from megspeech.objective.targets.speech import Targets
from tests.unit.megspeech.data.config import HELD, TRAIN
from tests.unit.megspeech.objective.targets.context import heard_run
from tests.unit.megspeech.objective.targets.speech import Root


def test_pool() -> None:
    """A sentence's words share the mean of those that have a target; a word without a sentence, or in a sentence
    of absent words, stays absent."""
    vectors = np.array([[0.0], [4.0], [np.nan], [np.nan], [10.0], [3.0], [8.0]], np.float16)
    pooled = Sentences(np.array([0, 0, 0, 1, 2, NO_SENTENCE, 3])).pool(Targets(vectors))
    assert pooled.present.tolist() == [True, True, True, False, True, False, True]
    values = pooled.vectors[:, 0].astype(np.float32)
    assert values[0] == values[1] == values[2]
    assert values[4] > values[6] > values[0]


def test_write(tmp_path: Path) -> None:
    run, held = heard_run(tmp_path), heard_run(tmp_path, HELD)
    root = cast("DataRoot", Root(tmp_path))
    source = Targets.directory(root, "src")
    for entry in (run, held):
        Targets(np.array([[1.0], [3.0]], np.float16)).save(source / f"{entry.name}.npy")
    directory = Sentences.write(root, "src")
    assert directory.name == "src" + SUFFIX
    assert [path.stem for path in directory.glob("*.npy")] == [Run(Path(TRAIN)).name]
    assert np.load(directory / f"{run.name}.npy").shape == (2, 1)
    assert Sentences.write(root, "src") == directory


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = heard_run(tmp_path)
    Targets(np.array([[1.0], [3.0]], np.float16)).save(tmp_path / "processed" / "speech" / "src" / f"{run.name}.npy")
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    monkeypatch.setattr("sys.argv", ["sentence", "src"])
    Sentences.main()
    assert (tmp_path / "processed" / "speech" / ("src" + SUFFIX) / f"{run.name}.npy").exists()

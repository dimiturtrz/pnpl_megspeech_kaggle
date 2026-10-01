from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from megspeech import epochs as module
from megspeech.epochs import DECIMATE, SFREQ, WINDOW, Epocher, Epochs, Window, main
from megspeech.runs import Run
from megspeech.vocabulary import Vocabulary

CHANNELS = 3
SECONDS = 4
WORDS = Vocabulary(("the", "a"))


def recording(root: Path) -> Run:
    run = Run(root / "derivatives" / "serialised" / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5")
    run.meg.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(run.meg, "w") as stored:
        stored["data"] = rng.normal(size=(CHANNELS, SECONDS * SFREQ)) * 1e-12
    run.events.parent.mkdir(parents=True)
    rows = [
        ("word", "The", 0.5),
        ("word", "zebra", 1.0),
        ("silence", None, 1.5),
        ("word", "a", 2.0),
        ("word", "the", SECONDS - 0.5),
    ]
    pd.DataFrame(rows, columns=["kind", "segment", "timemeg"]).to_csv(run.events, sep="\t", index=False)
    return run


def test_window_call() -> None:
    raw = np.ones((2, CHANNELS, WINDOW)) * 1e-12
    out = Window()(raw)
    assert out.shape == (2, CHANNELS, WINDOW // DECIMATE)
    assert out.dtype == np.float32
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-5)


def test_save(tmp_path: Path) -> None:
    epochs = Epochs(np.ones((1, CHANNELS, 2), np.float32), np.array([1]), np.array([0.5]))
    path = epochs.save(tmp_path / "x" / "e.npz")
    assert path.exists()


def test_load(tmp_path: Path) -> None:
    epochs = Epochs(np.ones((1, CHANNELS, 2), np.float32), np.array([1]), np.array([0.5]))
    loaded = Epochs.load(epochs.save(tmp_path / "e.npz"))
    assert loaded.x.dtype == np.float16
    assert loaded.y.tolist() == [1]


def test_epoch(tmp_path: Path) -> None:
    epochs = Epocher(tmp_path / "cache", WORDS).epoch(recording(tmp_path))
    assert epochs.y.tolist() == [0, 1]
    assert epochs.onset.tolist() == [0.5, 2.0]
    assert epochs.x.shape == (2, CHANNELS, WINDOW // DECIMATE)


def test_cut(tmp_path: Path) -> None:
    meg = np.arange(CHANNELS * 2 * WINDOW, dtype=np.float64).reshape(CHANNELS, -1) * 1e-12
    assert Epocher(tmp_path, WORDS).cut(meg, np.array([0, 10])).shape == (2, CHANNELS, WINDOW // DECIMATE)


def test_cached(tmp_path: Path) -> None:
    epocher = Epocher(tmp_path / "cache", WORDS)
    run = recording(tmp_path)
    first = epocher.cached(run)
    run.meg.unlink()
    assert np.array_equal(epocher.cached(run).x, first.x)


def test_labelled(tmp_path: Path) -> None:
    epocher = Epocher(tmp_path / "cache", WORDS)
    run = recording(tmp_path)
    epocher.cached(run)
    run.meg.unlink()
    y, onset = epocher.labelled(run)
    assert y.tolist() == [0, 1]
    assert onset.tolist() == [0.5, 2.0]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = recording(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: FakeRoot(tmp_path)))
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("sys.argv", ["epochs"])
    main()
    assert (tmp_path / "processed" / "epochs" / f"{run.name}.npz").exists()


class FakeRoot:
    def __init__(self, root: Path) -> None:
        self.root = root

    def raw(self) -> Path:
        return self.root

    def processed(self) -> Path:
        return self.root / "processed"


class FakePool:
    def __init__(self, _: int) -> None:
        pass

    def __enter__(self) -> "FakePool":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def map(self, function: Callable[[Run], Epochs], items: list[Run]) -> list[Epochs]:
        return [function(item) for item in items]

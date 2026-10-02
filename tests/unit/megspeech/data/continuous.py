from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data import continuous as module
from megspeech.data.continuous import DIRECTORY, NO_SENTENCE, RATE, WORDS, Continuous, Decimator, main
from megspeech.data.epochs import DECIMATE, HALF, SFREQ
from megspeech.data.runs import Run

CHANNELS = 3
SECONDS = 4
SAMPLES = SECONDS * RATE


def recording(root: Path, session: int = 1, rate: float = SFREQ) -> Run:
    run = Run(root / "derivatives" / "serialised" / f"sub-0_ses-{session}_task-Sherlock1_run-1_meg.h5")
    run.meg.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(session)
    with h5py.File(run.meg, "w") as stored:
        stored["data"] = (rng.normal(size=(CHANNELS, SECONDS * SFREQ)) * 1e-12).astype(np.float32)
        stored.attrs["sample_frequency"] = rate
    run.events.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("word", "The", 0.0, 0.5),
        ("phoneme", "dh_B", 0.0, 0.5),
        ("word", None, 0.0, 1.0),
        ("word", "null", 1.0, 2.0),
        ("word", "end", None, 3.0),
    ]
    columns = ["kind", "segment", "sentenceidx", "timemeg"]
    pd.DataFrame(rows, columns=columns).to_csv(run.events, sep="\t", index=False)
    return run


def test_path(tmp_path: Path) -> None:
    run = recording(tmp_path)
    assert Decimator(tmp_path / "cache").path(run.name) == tmp_path / "cache" / f"{run.name}.npy"


def test_ready(tmp_path: Path) -> None:
    decimator = Decimator(tmp_path / "a" / "cache")
    assert decimator.ready() == decimator.cache
    assert decimator.cache.is_dir()
    assert decimator.ready() == decimator.cache


def test_signal(tmp_path: Path) -> None:
    signal = Decimator(tmp_path).signal(recording(tmp_path))
    assert signal.shape == (CHANNELS, SECONDS * SFREQ // DECIMATE)
    assert signal.dtype == np.float16
    assert 0.1 < float(signal.astype(np.float32).std()) < 1
    with pytest.raises(ValueError, match="1000"):
        Decimator(tmp_path).signal(recording(tmp_path, session=2, rate=1000))


def test_signal_saturates(tmp_path: Path) -> None:
    run = recording(tmp_path)
    with h5py.File(run.meg, "r+") as stored:
        stored["data"][...] = 1.0
    signal = Decimator(tmp_path).signal(run)
    assert np.isfinite(signal).all()
    assert float(signal.max()) == HALF


def test_decimated() -> None:
    """The run's transform on any stretch: a stretch not a multiple of `DECIMATE` keeps its rounded-up length."""
    meg = np.full((CHANNELS, 2 * SFREQ + 3), 1e-12, dtype=np.float32)
    down = Decimator(Path()).decimated(meg)
    assert down.dtype == np.float16
    assert down.shape == (CHANNELS, Decimator(Path()).length(np.array([meg.shape[1]]))[0])
    assert np.allclose(down[:, RATE // 5 : -RATE // 5], 1.0, atol=1e-2)


def test_length() -> None:
    assert Decimator(Path()).length(np.array([0, 1, 5, 6, 1030])).tolist() == [0, 1, 1, 2, 206]


def test_save(tmp_path: Path) -> None:
    signal = np.arange(6, dtype=np.float16).reshape(2, 3)
    path = Decimator.save(tmp_path / "r.npy", signal)
    assert np.array_equal(np.load(path), signal)
    assert not list(tmp_path.glob("*.partial"))


def test_write(tmp_path: Path) -> None:
    decimator = Decimator(tmp_path / "cache")
    decimator.cache.mkdir()
    run = recording(tmp_path)
    path = decimator.write(run)
    assert np.load(path).shape == (CHANNELS, SAMPLES)
    run.meg.unlink()
    assert decimator.write(run) == path
    assert not list(decimator.cache.glob("*.partial"))


def test_words(tmp_path: Path) -> None:
    words = Decimator.words(recording(tmp_path))
    assert words["word"].tolist() == ["The", "null", "end"]
    assert words["onset"].tolist() == [0.5, 2.0, 3.0]
    assert words["sentence"].tolist() == [0, 1, NO_SENTENCE]


def test_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    runs = [recording(tmp_path, session) for session in (1, 2)]
    decimator = Decimator(tmp_path / "cache")
    table = decimator.build(runs)
    assert len(table) == 2 * 3
    assert (decimator.cache / WORDS).exists()
    assert all(decimator.path(run.name).exists() for run in runs)


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    runs = [recording(tmp_path, session) for session in (1, 2)]
    decimator = Decimator(tmp_path / "cache")
    decimator.build(runs)
    loaded = Continuous.load(decimator.cache, "cpu")
    assert loaded.meg.shape == (CHANNELS, 2 * SAMPLES)
    assert loaded.meg.dtype == torch.float16
    assert loaded.offsets.tolist() == [0, SAMPLES, 2 * SAMPLES]
    second = torch.from_numpy(np.load(decimator.path(runs[1].name)))
    assert torch.equal(loaded.meg[:, SAMPLES:], second)
    assert loaded.words["word"].tolist().count("null") == len(runs)
    assert loaded.words["at"].tolist() == [25, 100, 150, SAMPLES + 25, SAMPLES + 100, SAMPLES + 150]
    only = Continuous.load(decimator.cache, "cpu", [runs[1].name])
    assert only.names == (runs[1].name,)
    assert only.words["at"].tolist() == [25, 100, 150]


def test_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    runs = [recording(tmp_path, session) for session in (1, 2)]
    decimator = Decimator(tmp_path / "cache")
    decimator.build(runs)
    table = Continuous.table(decimator.cache, [runs[1].name, runs[0].name])
    assert table["run"].tolist() == [runs[1].name] * 3 + [runs[0].name] * 3
    assert table["sample"].tolist() == [25, 100, 150] * 2
    assert set(table["samples"]) == {SAMPLES}
    assert table.index.tolist() == list(range(6))
    assert Continuous.table(decimator.cache, [runs[0].name])["word"].tolist() == ["The", "null", "end"]


def test_inside() -> None:
    table = pd.DataFrame({"sample": [-1, 0, 150, 151], "samples": SAMPLES})
    assert Continuous.inside(table).tolist() == [False, True, True, False]
    assert Continuous.inside(table, length=10, offset=-5).tolist() == [False, False, True, True]


def test_windows() -> None:
    meg = torch.arange(2 * 100, dtype=torch.float32).reshape(2, 100)
    loaded = Continuous(meg, np.array([0, 100]), ("r",), pd.DataFrame({"at": [40, 0]}))
    windows = loaded.windows()
    assert len(windows) == len(loaded.words)
    assert windows(torch.tensor([0])).shape == (1, 2, RATE)
    assert windows.at.tolist() == [40, 0]


def test_gigabytes() -> None:
    meg = torch.zeros((CHANNELS, 1000), dtype=torch.float16)
    loaded = Continuous(meg, np.array([0, 1000]), ("r",), pd.DataFrame())
    assert loaded.gigabytes == pytest.approx(CHANNELS * 1000 * 2 / 1e9)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = recording(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: FakeRoot(tmp_path)))
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("sys.argv", ["continuous"])
    main()
    assert (tmp_path / "processed" / DIRECTORY / f"{run.name}.npy").exists()


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

    def map(self, function: Callable[[Run], Path], items: list[Run]) -> list[Path]:
        return [function(item) for item in items]

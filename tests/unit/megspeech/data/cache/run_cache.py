from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from megspeech.data.cache import run_cache as module
from megspeech.data.cache.run_cache import DIRECTORY, NO_SENTENCE, TAIL, WORD_TABLE, RunCache, main
from megspeech.data.constants import DECIMATE, HALF, RATE, SFREQ
from megspeech.data.raw.runs import Run

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
        ("word", "The", 0.0, 0.5, 0.3),
        ("phoneme", "dh_B", 0.0, 0.5, 2.0),
        ("word", None, 0.0, 1.0, 2.0),
        ("word", "null", 1.0, 2.0, 0.4),
        ("word", "end", None, 3.0, None),
    ]
    columns = ["kind", "segment", "sentenceidx", "timemeg", "duration"]
    pd.DataFrame(rows, columns=columns).to_csv(run.events, sep="\t", index=False)
    return run


def test_path(tmp_path: Path) -> None:
    run = recording(tmp_path)
    assert RunCache(tmp_path / "cache").path(run.name) == tmp_path / "cache" / f"{run.name}.npy"


def test_under(tmp_path: Path) -> None:
    assert RunCache.under(tmp_path) == RunCache(tmp_path / "continuous50")
    assert RunCache.under(tmp_path, 4) == RunCache(tmp_path / "continuous62.5", 4)


def test_rate() -> None:
    assert RunCache(Path()).rate == RATE
    assert RunCache(Path(), 2).rate == SFREQ / 2


def test_ready(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "a" / "cache")
    assert cache.ready() == cache.cache
    assert cache.cache.is_dir()
    assert cache.ready() == cache.cache


def test_signal(tmp_path: Path) -> None:
    signal = RunCache(tmp_path).signal(recording(tmp_path))
    assert signal.shape == (CHANNELS, SECONDS * SFREQ // DECIMATE)
    assert signal.dtype == np.float16
    assert 0.1 < float(signal.astype(np.float32).std()) < 1
    with pytest.raises(ValueError, match="1000"):
        RunCache(tmp_path).signal(recording(tmp_path, session=2, rate=1000))


def test_signal_saturates(tmp_path: Path) -> None:
    run = recording(tmp_path)
    with h5py.File(run.meg, "r+") as stored:
        stored["data"][...] = 1.0
    signal = RunCache(tmp_path).signal(run)
    assert np.isfinite(signal).all()
    assert float(signal.max()) == HALF


def test_decimated() -> None:
    """The run's transform on any stretch: a stretch not a multiple of `DECIMATE` keeps its rounded-up length."""
    meg = np.full((CHANNELS, 2 * SFREQ + 3), 1e-12, dtype=np.float32)
    down = RunCache(Path()).decimated(meg)
    assert down.dtype == np.float16
    assert down.shape == (CHANNELS, RunCache(Path()).length(np.array([meg.shape[1]]))[0])
    assert np.allclose(down[:, RATE // 5 : -RATE // 5], 1.0, atol=1e-2)


def test_length() -> None:
    assert RunCache(Path()).length(np.array([0, 1, 5, 6, 1030])).tolist() == [0, 1, 1, 2, 206]


def test_save(tmp_path: Path) -> None:
    signal = np.arange(6, dtype=np.float16).reshape(2, 3)
    path = RunCache.save(tmp_path / "r.npy", signal)
    assert np.array_equal(np.load(path), signal)
    assert not list(tmp_path.glob("*.partial"))


def test_write(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.cache.mkdir()
    path = cache.write("r", lambda: np.ones((2, 3), dtype=np.float16))
    assert path == cache.path("r")
    assert np.load(path).shape == (2, 3)
    assert cache.write("r", lambda: pytest.fail("a cached run is produced again")) == path
    assert not list(cache.cache.glob("*.partial"))


def test_cache_run(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.cache.mkdir()
    run = recording(tmp_path)
    path = cache.cache_run(run)
    assert np.load(path).shape == (CHANNELS, SAMPLES)
    run.meg.unlink()
    assert cache.cache_run(run) == path


def test_staged(tmp_path: Path) -> None:
    final = tmp_path / "words.tsv"
    with RunCache.staged(final) as staging:
        assert staging == tmp_path / "words.tsv.partial"
        staging.write_text("x")
        assert not final.exists()
    assert final.read_text() == "x"
    assert not staging.exists()


def test_words(tmp_path: Path) -> None:
    words = RunCache.words(recording(tmp_path))
    assert words["word"].tolist() == ["The", "null", "end"]
    assert words["onset"].tolist() == [0.5, 2.0, 3.0]
    assert words["sentence"].tolist() == [0, 1, NO_SENTENCE]
    assert words["end"].tolist() == pytest.approx([0.8 + TAIL, 2.4 + TAIL, 3.0 + TAIL])


def test_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    runs = [recording(tmp_path, session) for session in (1, 2)]
    cache = RunCache(tmp_path / "cache")
    table = cache.build(runs)
    assert len(table) == 2 * 3
    assert (cache.cache / WORD_TABLE).exists()
    assert all(cache.path(run.name).exists() for run in runs)


def test_publish(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.ready()
    table = pd.DataFrame({"run": ["a"], "onset": [0.5]})
    assert cache.publish(table) is table
    assert pd.read_csv(cache.cache / WORD_TABLE, sep="	").equals(table)
    assert not list(cache.cache.glob("*.partial"))


def test_published(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.ready()
    cache.publish(pd.DataFrame({"run": ["a", "b"], "word": ["null", "None"]}))
    assert cache.published()["word"].tolist() == ["null", "None"]


def test_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    runs = [recording(tmp_path, session) for session in (1, 2)]
    cache = RunCache(tmp_path / "cache")
    cache.build(runs)
    table = cache.table([runs[1].name, runs[0].name])
    assert table["run"].tolist() == [runs[1].name] * 3 + [runs[0].name] * 3
    assert table["sample"].tolist() == [25, 100, 150] * 2
    assert set(table["samples"]) == {SAMPLES}
    assert table["stop"].tolist() == [90, 170, SAMPLES] * 2
    assert table["opening"].tolist() == table["sample"].tolist()
    assert table.index.tolist() == list(range(6))
    assert cache.table([runs[0].name])["word"].tolist() == ["The", "null", "end"]
    faster = RunCache(cache.cache, 2).table([runs[0].name])
    assert faster["sample"].tolist() == [62, 250, 375]
    assert faster["stop"].tolist() == [SAMPLES] * 3


def test_inside() -> None:
    table = pd.DataFrame({"sample": [-1, 0, 150, 151], "samples": SAMPLES})
    assert RunCache.inside(table).tolist() == [False, True, True, False]
    assert RunCache.inside(table, length=10, offset=-5).tolist() == [False, False, True, True]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = recording(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: FakeRoot(tmp_path)))
    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("sys.argv", ["run_cache"])
    main()
    assert (tmp_path / "processed" / DIRECTORY.format(rate=RATE) / f"{run.name}.npy").exists()


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

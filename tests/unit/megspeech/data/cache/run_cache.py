import os
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.cache import run_cache as module
from megspeech.data.cache.alignment import RUN_LEVELS, Geometry, Projection
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.cache.run_cache import DIRECTORY, TAIL, WORD_TABLE, DeviceRecording, RunCache, main
from megspeech.data.constants import DECIMATE, HALF, RATE, SFREQ
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import Grouping

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
    assert RunCache.under(tmp_path, cleaning=Cleaning.ICA) == RunCache(
        tmp_path / "continuous50-ica", DECIMATE, Cleaning.ICA
    )


def test_cleaner(tmp_path: Path) -> None:
    meg = np.ones((CHANNELS, SAMPLES), dtype=np.float32)
    assert RunCache(tmp_path).cleaner(lambda: tmp_path / "missing.h5")(meg) is meg


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
    source = tmp_path / "source.npz"
    source.touch()
    os.utime(source, (path.stat().st_mtime - 1,) * 2)
    assert cache.write("r", lambda: pytest.fail("a source older than its cache is produced again"), source) == path
    os.utime(source, (path.stat().st_mtime + 1,) * 2)
    cache.write("r", lambda: np.zeros((4, 5), dtype=np.float16), source)
    assert np.load(path).shape == (4, 5)


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
    cache.published()["word"] = "changed"
    assert cache.published()["word"].tolist() == ["null", "None"]
    cache.publish(pd.DataFrame({"run": ["c"], "word": ["new"]}))
    stamp = (cache.cache / WORD_TABLE).stat().st_mtime_ns + 1
    os.utime(cache.cache / WORD_TABLE, ns=(stamp, stamp))
    assert cache.published()["word"].tolist() == ["new"]


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
    assert table["following"].tolist() == table["stop"].tolist()
    assert table["piece_stop"].tolist() == table["stop"].tolist()
    assert table["piece_opening"].tolist() == table["opening"].tolist()
    assert table["piece_following"].tolist() == table["following"].tolist()
    assert table.index.tolist() == list(range(6))
    assert cache.table([runs[0].name])["word"].tolist() == ["The", "null", "end"]
    faster = RunCache(cache.cache, 2).table([runs[0].name])
    assert faster["sample"].tolist() == [62, 250, 375]
    assert faster["stop"].tolist() == [SAMPLES] * 3


def test_inside() -> None:
    table = pd.DataFrame({"sample": [-1, 0, 150, 151], "samples": SAMPLES})
    assert RunCache.inside(table).tolist() == [False, True, True, False]
    assert RunCache.inside(table, length=10, offset=-5).tolist() == [False, False, True, True]


def test_carried() -> None:
    table = pd.DataFrame({"word": ["the", "zebra", "a"], "sample": [0, 10, 20], "samples": [SAMPLES] * 3})
    assert RunCache.carried(table, Grouping.SENTENCE)["word"].tolist() == ["the", "zebra", "a"]
    assert RunCache.carried(table, Grouping.GAP)["word"].tolist() == ["the", "a"]
    assert RunCache.carried(table, Grouping.SENTENCE)["target"].tolist()[1] == IGNORE


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

    def processed(self, *parts: str) -> Path:
        return self.root.joinpath("processed", *parts)


class FakePool:
    def __init__(self, _: int, **__: object) -> None:
        pass

    def __enter__(self) -> "FakePool":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def map(self, function: Callable[[Run], Path], items: list[Run]) -> list[Path]:
        return [function(item) for item in items]


NAMES = ("a", "b")


def built(tmp_path: Path) -> RunCache:
    """Two cached runs of `SAMPLES` with three words each, at 0.5, 2 and 3 s."""
    cache = RunCache(tmp_path / "cache")
    cache.ready()
    rng = np.random.default_rng(0)
    for name in NAMES:
        RunCache.save(cache.path(name), (rng.normal(size=(CHANNELS, SAMPLES)) * 3 + 7).astype(np.float16))
    heard = ((0.5, "The"), (2.0, "null"), (3.0, "end"))
    rows = [(name, onset, word, 0, 4.0) for name in NAMES for onset, word in heard]
    cache.publish(pd.DataFrame(rows, columns=["run", "onset", "word", "sentence", "end"]))
    return cache


def test_load(tmp_path: Path) -> None:
    cache = built(tmp_path)
    loaded = DeviceRecording.load(cache, "cpu")
    assert loaded.meg.shape == (CHANNELS, 2 * SAMPLES)
    assert loaded.meg.dtype == torch.float16
    assert loaded.offsets.tolist() == [0, SAMPLES, 2 * SAMPLES]
    assert loaded.names == NAMES
    assert torch.equal(loaded.meg[:, SAMPLES:], torch.from_numpy(np.load(cache.path("b"))))
    assert loaded.words["word"].tolist().count("null") == len(NAMES)
    assert loaded.words["at"].tolist() == [25, 100, 150, SAMPLES + 25, SAMPLES + 100, SAMPLES + 150]
    assert loaded.words["start"].tolist() == [0] * 3 + [SAMPLES] * 3
    only = DeviceRecording.load(cache, "cpu", ["b"])
    assert only.names == ("b",)
    assert only.words["at"].tolist() == [25, 100, 150]
    assert DeviceRecording.load(cache, "cpu", ["b"]) is only
    again = DeviceRecording.load(cache, "cpu")
    assert again is not loaded
    assert torch.equal(again.meg, loaded.meg)


def test_read(tmp_path: Path) -> None:
    """Every call reads afresh."""
    cache = built(tmp_path)
    first, second = (DeviceRecording.read(cache, "cpu", ("b",)) for _ in range(2))
    assert first is not second
    assert torch.equal(first.meg, second.meg)
    assert first.words.equals(second.words)


def subspace() -> Projection:
    basis = torch.linalg.qr(torch.randn(CHANNELS, 2, generator=torch.Generator().manual_seed(0)))[0].T.contiguous()
    return Projection(torch.full((CHANNELS, 1), 7.0), torch.full((CHANNELS, 1), 4.0), basis)


def test_load_projected(tmp_path: Path) -> None:
    cache = built(tmp_path)
    projection = subspace()
    loaded = DeviceRecording.load(cache, "cpu", geometry=Geometry(projection=projection))
    full = DeviceRecording.load(cache, "cpu")
    assert loaded.meg.shape == (projection.rank, 2 * SAMPLES)
    assert loaded.meg.dtype == torch.float16
    assert loaded.offsets.tolist() == full.offsets.tolist()
    assert loaded.words.equals(full.words)
    for begin, name in zip((0, SAMPLES), NAMES, strict=True):
        run = torch.from_numpy(np.load(cache.path(name)))
        expected = projection.project(run).half()
        assert torch.allclose(loaded.meg[:, begin : begin + SAMPLES].float(), expected.float(), atol=1e-2)


def test_load_levelled(tmp_path: Path) -> None:
    """Each run loads at its own levels: a run's statistics never reach another run's rows."""
    cache = built(tmp_path)
    loaded = DeviceRecording.load(cache, "cpu", geometry=Geometry(levels=RUN_LEVELS))
    assert loaded.meg.dtype == torch.float16
    for begin, name in zip((0, SAMPLES), NAMES, strict=True):
        own = RUN_LEVELS.level(torch.from_numpy(np.load(cache.path(name))))
        assert torch.allclose(loaded.meg[:, begin : begin + SAMPLES].float(), own, atol=1e-2)


def test_windows() -> None:
    meg = torch.arange(2 * 100, dtype=torch.float32).reshape(2, 100)
    words = pd.DataFrame({"at": [40, 0], "start": [0, 0], "stop": [90, 50], "opening": [30, 0], "following": [60, 50]})
    loaded = DeviceRecording(meg, np.array([0, 100]), ("r",), words)
    windows = loaded.windows()
    assert len(windows) == len(loaded.words)
    assert windows(torch.tensor([0])).shape == (1, 2, RATE)
    assert windows.at.tolist() == [40, 0]
    assert windows.end.tolist() == [90, 50]
    assert windows.begin is not None
    assert windows.begin.tolist() == [30, 0]
    assert windows.following is not None
    assert windows.following.tolist() == [60, 50]
    assert loaded.windows(2 * RATE)(torch.tensor([0]))[0, :, RATE:].eq(0).all()


def test_samples() -> None:
    loaded = DeviceRecording(torch.zeros(2, 10), np.array([0, 10]), ("r",), pd.DataFrame({"at": [4, 0]}))
    samples = loaded.samples(loaded.words["at"] + 1)
    assert samples.dtype == torch.int64
    assert samples.tolist() == [5, 1]


def test_gigabytes() -> None:
    meg = torch.zeros((CHANNELS, 1000), dtype=torch.float16)
    loaded = DeviceRecording(meg, np.array([0, 1000]), ("r",), pd.DataFrame())
    assert loaded.gigabytes == pytest.approx(CHANNELS * 1000 * 2 / 1e9)


def test_bounded() -> None:
    cache = RunCache(Path("unused"))
    rate = cache.rate
    table = pd.DataFrame(
        {
            "run": "a",
            "piece": [0, 0, 1],
            "sample": [0, round(rate), round(3 * rate)],
            "samples": round(10 * rate),
            "end": [4.0, 4.0, 5.0],
        }
    )
    cache.bounded(table, "piece_", "end", ["piece"])
    assert table["piece_opening"].tolist() == [0, 0, round(3 * rate)]
    assert table["piece_stop"].tolist() == [round(4 * rate)] * 2 + [round(5 * rate)]
    assert table["piece_following"].tolist() == [round(rate), round(4 * rate), round(5 * rate)]


def test_table_pieces(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.ready()
    nine = pd.DataFrame(
        {
            "run": "a",
            "onset": np.arange(9) * 0.5,
            "word": "w",
            "sentence": 0,
            "end": 4.5 + 9 + TAIL,
            "finish": np.arange(9) * 0.5 + 0.3,
        }
    )
    cache.publish(nine)
    cache.write("a", lambda: np.zeros((1, 30 * round(cache.rate)), dtype=np.float16))
    table = cache.table(["a"])
    assert table["piece"].tolist() == [0] * 5 + [1] * 4
    assert table["piece_end"].tolist() == pytest.approx([2.3 + TAIL] * 5 + [4.3 + TAIL] * 4)
    assert table["piece_opening"].tolist() == [0] * 5 + [round(2.5 * cache.rate)] * 4
    assert (table["piece_stop"] < table["stop"]).all()

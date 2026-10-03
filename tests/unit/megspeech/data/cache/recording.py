from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.cache.recording import DeviceRecording
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import RATE

CHANNELS = 3
SAMPLES = 4 * RATE
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
    scaled = DeviceRecording.load(cache, "cpu", standardised=True).meg.float()
    for begin in (0, SAMPLES):
        run = scaled[:, begin : begin + SAMPLES]
        assert run.median(dim=1).values.abs().max() < 0.05
        assert torch.allclose(run.quantile(0.75, dim=1) - run.quantile(0.25, dim=1), torch.ones(CHANNELS), atol=0.05)


def test_standardise() -> None:
    offset, gain = torch.tensor([[5.0], [-3.0]]), torch.tensor([[2.0], [0.5]])
    signal = torch.linspace(-1, 1, 101).repeat(2, 1)
    run = signal * gain + offset
    DeviceRecording.standardise(run)
    assert torch.allclose(run, signal, atol=1e-5)
    flat = torch.cat([signal[:1] * 2.0 + 1.0, torch.full((1, 101), 4.0)])
    DeviceRecording.standardise(flat)
    assert torch.isfinite(flat).all()
    assert torch.equal(flat[1], torch.zeros(101))
    assert torch.allclose(flat[0], signal[0], atol=1e-5)


def test_windows() -> None:
    meg = torch.arange(2 * 100, dtype=torch.float32).reshape(2, 100)
    words = pd.DataFrame({"at": [40, 0], "start": [0, 0], "stop": [90, 50], "opening": [30, 0]})
    loaded = DeviceRecording(meg, np.array([0, 100]), ("r",), words)
    windows = loaded.windows()
    assert len(windows) == len(loaded.words)
    assert windows(torch.tensor([0])).shape == (1, 2, RATE)
    assert windows.at.tolist() == [40, 0]
    assert windows.end.tolist() == [90, 50]
    assert windows.begin is not None
    assert windows.begin.tolist() == [30, 0]
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

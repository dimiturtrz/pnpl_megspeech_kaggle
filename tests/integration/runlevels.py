"""Per-run levels from the data config to the Kaggle test's windows: an unlabelled archive is read at its own
robust levels, never the identity, and the same archive at another session's gain and offset reads alike."""

from pathlib import Path

import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import AS_RECORDED, RUN_LEVELS, Alignment, Geometry
from megspeech.data.cache.kaggle import KaggleCache
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.constants import RATE
from tests.unit.megspeech.data.cache.kaggle import kaggle

CPU = "cpu"
GAIN = np.array([5.0, 0.25, 40.0])[:, None]
OFFSET = np.array([3.0, -1.0, 0.0])[:, None]


def levelled(root: Path) -> Geometry:
    return DataConfig(levelled=True).geometry(DataRoot(root), CPU)


def loaded(cache: KaggleCache, geometry: Geometry) -> DeviceRecording:
    return DeviceRecording.load(cache.cache, CPU, [archive.name for archive in cache.archives()], geometry=geometry)


def every(cache: KaggleCache, geometry: Geometry) -> torch.Tensor:
    windows = cache.windows(CPU, RATE, geometry=geometry)
    return windows(torch.arange(len(windows))).float()


def regained(cache: KaggleCache) -> None:
    """`cache`'s cached archives rewritten at `GAIN` and `OFFSET` on every channel, their padding left silent."""
    for archive in cache.archives():
        path = cache.cache.path(archive.name)
        meg = np.load(path)
        np.save(path, np.where(np.abs(meg).max(0) > 0, meg * GAIN + OFFSET, 0.0).astype(meg.dtype))


def test_kaggle_archives_read_their_own_levels(tmp_path: Path) -> None:
    """Each unlabelled archive loads at the levels of its own MEG: median 0 and range 1 on its signal, unlike the
    archive as recorded, and its windows move with it."""
    cache = kaggle(tmp_path / "raw")
    cache.build()
    geometry = levelled(tmp_path)
    assert geometry.levels is RUN_LEVELS
    recorded, read = loaded(cache, AS_RECORDED), loaded(cache, geometry)
    for begin, end in zip(read.offsets[:-1], read.offsets[1:], strict=True):
        signal = read.meg[:, begin:end].float()
        assert torch.allclose(signal, RUN_LEVELS.level(recorded.meg[:, begin:end]), atol=1e-2)
        median, spread = Alignment.robust(signal[:, signal.abs().amax(0) > 0])
        assert torch.allclose(median, torch.zeros_like(median), atol=0.02)
        assert torch.allclose(spread, torch.ones_like(spread), atol=0.02)
        assert not torch.allclose(signal, recorded.meg[:, begin:end].float(), atol=0.1)
    assert not torch.allclose(every(cache, geometry), every(cache, AS_RECORDED), atol=0.1)


def test_kaggle_archives_at_another_gain_read_alike(tmp_path: Path) -> None:
    cache = kaggle(tmp_path / "raw")
    cache.build()
    other = kaggle(tmp_path / "other" / "raw")
    other.build()
    regained(other)
    assert not torch.allclose(every(cache, AS_RECORDED), every(other, AS_RECORDED), atol=0.1)
    assert torch.allclose(every(cache, levelled(tmp_path)), every(other, levelled(tmp_path)), atol=2e-2)

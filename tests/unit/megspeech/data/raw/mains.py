import json
from pathlib import Path

import mne
import numpy as np
import pytest

from megspeech.data.raw.mains import Mains

SFREQ = 1000.0


def sidecar(tmp_path: Path, recorded: object) -> Path:
    path = tmp_path / "sub-01_meg.json"
    path.write_text(json.dumps({"PowerLineFrequency": recorded}))
    return path


def test_of(tmp_path: Path) -> None:
    assert Mains.of(sidecar(tmp_path, 60.0)) == Mains(60.0)
    assert Mains.of(sidecar(tmp_path, 50)) == Mains(50.0)
    assert Mains.of(sidecar(tmp_path, "n/a"), 50.0) == Mains(50.0)
    with pytest.raises(ValueError, match="PowerLineFrequency"):
        Mains.of(sidecar(tmp_path, "n/a"))


def test_harmonics() -> None:
    assert Mains(60.0).harmonics == [60.0, 120.0]
    assert Mains(50.0).harmonics == [50.0, 100.0]


def test_notch() -> None:
    times = np.arange(20 * SFREQ) / SFREQ
    signal = np.sin(2 * np.pi * 50 * times) + np.sin(2 * np.pi * 10 * times)
    raw = mne.io.RawArray(signal[None], mne.create_info(["MEG 001"], SFREQ, "mag"), verbose="error")
    notched = Mains(50.0).notch(raw).get_data()[0]
    assert np.std(notched - np.sin(2 * np.pi * 10 * times)) < 0.05 * np.std(signal)

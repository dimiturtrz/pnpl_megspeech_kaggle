from pathlib import Path

import h5py
import numpy as np

from megspeech.data.cache.cleaning import REJECT, SEPARATOR, SPAN, Cleaning, Ica, Unchanged
from megspeech.data.constants import SFREQ

SENSORS = 102
SOURCES = 30
SECONDS = 60
BEAT = 1.1
QRS = 0.04
NOISE = 1e-15


def layout() -> Ica:
    """A 306-channel Vectorview-like layout: per sensor one magnetometer then two gradiometers."""
    names = tuple(f"MEG{sensor:03d}{kind}" for sensor in range(1, SENSORS + 1) for kind in (1, 2, 3))
    types = tuple(("mag", "grad", "grad")[int(name[-1]) - 1] for name in names)
    return Ica(names, types)


def recording() -> tuple[np.ndarray, np.ndarray]:
    """A 1.1 s periodic, QRS-shaped spike train, and low-rank brain-like activity plus that train at ten times its
    amplitude on every magnetometer."""
    rng = np.random.default_rng(0)
    samples = SECONDS * SFREQ
    time = np.arange(samples) / SFREQ
    sources = np.cumsum(rng.normal(size=(SOURCES, samples)), axis=1)
    sources -= sources.mean(1, keepdims=True)
    sources /= sources.std(1, keepdims=True)
    channels = 3 * SENSORS
    brain = 1e-13 * rng.normal(size=(channels, SOURCES)) @ sources + NOISE * rng.normal(size=(channels, samples))
    phase = np.mod(time, BEAT)
    beat = np.clip(1 - np.abs(phase - BEAT / 2) / (QRS / 2), 0, None)
    beat = (beat - beat.mean()) / beat.std()
    topography = np.where(np.arange(channels) % 3 == 0, 1.0, 0.1) * rng.uniform(0.5, 1.5, channels)
    return beat, brain + 1e-12 * topography[:, None] * beat


def test_of(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as stored:
        stored.attrs["channel_names"] = f"{SEPARATOR} ".join(("MEG0111", "MEG0112"))
        stored.attrs["channel_types"] = f"{SEPARATOR} ".join(("mag", "grad"))
    assert Ica.of(path) == Ica(("MEG0111", "MEG0112"), ("mag", "grad"))


def test_stretches() -> None:
    span = round(SPAN * SFREQ)
    meg = np.arange(2 * (2 * span + 1)).reshape(2, -1)
    stretched = Ica.stretches(meg)
    assert stretched.shape == (2, 2, span)
    assert stretched[1, 1, 0] == meg[1, span]


def test_rejection() -> None:
    span = round(SPAN * SFREQ)
    stretches = 5
    highpassed = np.zeros((3, stretches * span + 1))
    highpassed[:, ::span][:, :stretches] = [[1.0], [2.0], [3.0]]
    highpassed[0, span] = 100.0
    rejection = Ica(("MEG0111", "MEG0112", "MEG0113"), ("mag", "grad", "grad"))
    assert rejection.rejection(highpassed) == {"mag": REJECT * 1.0, "grad": REJECT * 3.0}


def test_call() -> None:
    beat, meg = recording()
    cleaned = layout()(meg.astype(np.float32))
    assert cleaned.shape == meg.shape
    assert cleaned.dtype == np.float32
    assert np.linalg.norm(cleaned @ beat) < 0.1 * np.linalg.norm(meg @ beat)


def test_cleaner(tmp_path: Path) -> None:
    meg = np.ones((2, 3), dtype=np.float32)
    assert Cleaning.NONE.cleaner(lambda: tmp_path / "missing.h5")(meg) is meg
    assert isinstance(Cleaning.NONE.cleaner(lambda: tmp_path), Unchanged)
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as stored:
        stored.attrs["channel_names"] = "MEG0111"
        stored.attrs["channel_types"] = "mag"
    assert Cleaning.ICA.cleaner(lambda: path) == Ica(("MEG0111",), ("mag",))

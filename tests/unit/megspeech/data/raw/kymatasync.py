import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from megspeech.data.raw.kymatasync import DRIFT_BOUNDS, FALLBACK, RATE, Alignment, Clock, Envelope

CLIP = 40
WINDOW = 10
DRIFT = 600e-6
RUN = 130
START = 20.0
OFFSET = 0.025


def burst(seed: int, seconds: int) -> np.ndarray:
    """A smooth non-negative envelope with structure at every scale a correlation can lock to."""
    rng = np.random.default_rng(seed)
    noise = np.convolve(rng.standard_normal(seconds * RATE), np.ones(25) / 25, mode="same")
    return np.abs(noise) + 0.05


def reference() -> Alignment:
    return Alignment(Envelope.of(burst(0, CLIP), RATE), window=WINDOW * RATE)


def played(alignment: Alignment, starts: list[float], drift: float) -> Envelope:
    """A run's audio channel: the clip stretched by `drift` at each start over weak noise."""
    run = burst(1, RUN) * 0.05
    clip = alignment.reference.samples - alignment.reference.samples.min()
    grid = np.arange(int(len(clip) * (1 + drift)))
    stretched = np.asarray(np.interp(grid / (1 + drift), np.arange(len(clip)), clip))
    for start in starts:
        begin = round(start * RATE)
        run[begin : begin + len(stretched)] += stretched
    return Envelope.of(run, RATE)


def test_start() -> None:
    assert Clock(10.0, 0.02, 0.0).start == pytest.approx(10.02)


def test_place() -> None:
    events = pd.DataFrame({"timemeg": [0.0, 100.0], "duration": [1.0, 2.0]})
    placed = Clock(10.0, 0.02, 500e-6).place(events)
    assert placed["timemeg"].tolist() == pytest.approx([10.02, 10.02 + 100.05])
    assert placed["duration"].tolist() == pytest.approx([1.0005, 2.001])
    assert events["timemeg"].tolist() == [0.0, 100.0]


def test_fallen(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        clock = Clock.fallen(7.0, "no cover")
    assert clock == Clock(7.0, FALLBACK.offset, FALLBACK.drift, fitted=False)
    assert "no cover" in caplog.text


def test_summary() -> None:
    clocks = [Clock(0.0, 0.020, 500e-6), Clock(0.0, 0.022, 540e-6), Clock(0.0, 0.024, 580e-6), Clock.fallen(1.0, "x")]
    assert Clock.summary(clocks) == "3 fitted, 1 fell back; offset ms 20.0/22.0/24.0 drift ppm 500/540/580"


def test_envelope_of() -> None:
    envelope = Envelope.of(np.sin(np.arange(4 * RATE) / 7.0), RATE)
    assert envelope.samples.mean() == pytest.approx(0, abs=1e-9)
    assert envelope.samples.std() == pytest.approx(1)
    assert len(Envelope.of(np.sin(np.arange(2 * 4000) / 7.0), 4000).samples) == 2 * RATE


def test_span() -> None:
    assert Envelope(np.arange(10.0)).span(2, 5).samples.tolist() == [2.0, 3.0, 4.0]


def test_correlation() -> None:
    signal = Envelope.of(burst(3, 20), RATE)
    match = signal.correlation(signal.span(4000, 6000))
    assert len(match) == len(signal.samples) - 2000 + 1
    assert int(np.argmax(match)) == 4000
    assert match.max() == pytest.approx(1, abs=1e-6)


def test_seconds() -> None:
    assert reference().seconds == CLIP


def test_locate() -> None:
    alignment = reference()
    starts = [10.0, 70.0]
    found = alignment.locate(played(alignment, starts, DRIFT))
    middle = FALLBACK.offset + FALLBACK.drift * CLIP / 2
    assert found == pytest.approx([start - middle for start in starts], abs=0.03)


def test_lags() -> None:
    alignment = reference()
    centres, lags = alignment.lags(played(alignment, [START + OFFSET], DRIFT), START)
    assert centres.tolist() == [5.0, 15.0, 25.0]
    assert lags == pytest.approx(OFFSET + DRIFT * centres, abs=2 / RATE)


def test_clock() -> None:
    alignment = reference()
    clock = alignment.clock(played(alignment, [START + OFFSET], DRIFT), START)
    assert clock.fitted
    assert clock.offset == pytest.approx(OFFSET, abs=0.005)
    assert clock.drift == pytest.approx(DRIFT, abs=0.5 * DRIFT)


def test_clock_out_of_bounds(caplog: pytest.LogCaptureFixture) -> None:
    alignment = reference()
    with caplog.at_level(logging.WARNING):
        clock = alignment.clock(played(alignment, [START + OFFSET], 2 * DRIFT_BOUNDS[1]), START)
    assert clock == Clock(START, FALLBACK.offset, FALLBACK.drift, fitted=False)
    assert "ppm" in caplog.text


def test_clock_unrelated_audio() -> None:
    alignment = reference()
    assert not alignment.clock(Envelope.of(burst(9, RUN), RATE), START).fitted


def test_clock_uncovered() -> None:
    alignment = reference()
    recorded = played(alignment, [START + OFFSET], DRIFT)
    assert not alignment.clock(recorded, RUN - 5).fitted
    assert not alignment.clock(recorded, 0.0).fitted


def test_alignment_of(tmp_path: Path) -> None:
    rate = 8000
    wavfile.write(tmp_path / "a.wav", rate, np.random.default_rng(0).standard_normal((3 * rate, 2)))
    assert Alignment.of(tmp_path / "a.wav").seconds == 3


def test_alignment_of_mono(tmp_path: Path) -> None:
    rate = 8000
    wavfile.write(tmp_path / "a.wav", rate, np.random.default_rng(0).standard_normal(3 * rate).astype(np.float32))
    assert Alignment.of(tmp_path / "a.wav").seconds == 3


def test_locate_plays() -> None:
    alignment = reference()
    found = alignment.locate(played(alignment, [10.0, 70.0], DRIFT), 1)
    assert len(found) == 1


def test_clock_offsets() -> None:
    alignment = reference()
    recorded = played(alignment, [START + 0.1], DRIFT)
    assert not alignment.clock(recorded, START).fitted
    assert alignment.clock(recorded, START, (-0.2, 0.2)).fitted

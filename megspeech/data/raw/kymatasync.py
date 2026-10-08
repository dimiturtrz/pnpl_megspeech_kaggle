"""Kymata-SOTO's play clock: where the played clip sits in the recording, fitted from the stimulus audio the recording
carries on `MISC006` rather than trusted from the start trigger.

Measured over 160 plays: the audio starts `FALLBACK.offset` after the trigger, and the stimulus machine's clock runs
`FALLBACK.drift` slow against the MEG's, so a clip time `t` lands at `start + (1 + drift) * t`.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from jaxtyping import Float
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

LOG = logging.getLogger(__name__)

RATE = 1000
"""Samples per second of every envelope: the MEG's own rate."""
CUTOFF = 30
"""Hz above which the rectified audio is smoothed away, leaving its envelope."""
FILTER_ORDER = 4
WINDOW = 50 * RATE
"""Samples of clip correlated at once when fitting the drift."""
LAG_LOW = -200
LAG_HIGH = 400
"""Milliseconds around a play's nominal start searched for each window (drift alone reaches ~220 ms by clip end)."""
PLAYS_PER_RUN = 2
VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class Clock:
    """One play's clip-to-recording map: `nominal` seconds (its trigger, or the audio's coarse find) plus `offset`
    is where clip time 0 plays, every clip second then lasting `1 + drift` recorded ones."""

    nominal: float
    offset: float
    drift: float
    fitted: bool = True

    @property
    def start(self) -> float:
        return self.nominal + self.offset

    def place(self, events: pd.DataFrame) -> pd.DataFrame:
        """`events` from clip time into recording time: `timemeg` mapped, `duration` stretched."""
        scale = 1 + self.drift
        return events.assign(timemeg=self.start + scale * events["timemeg"], duration=scale * events["duration"])

    @classmethod
    def fallen(cls, nominal: float, reason: str) -> "Clock":
        """The constant model at `nominal`, `reason` logged for why the play's own fit was not used."""
        LOG.warning("play at %.3f s: %s; using the constant clock", nominal, reason)
        return cls(nominal, FALLBACK.offset, FALLBACK.drift, fitted=False)

    @classmethod
    def summary(cls, clocks: list["Clock"]) -> str:
        """Min / median / max of the fitted offsets (ms) and drifts (ppm), and how many plays fell back."""
        fitted = [clock for clock in clocks if clock.fitted]
        offsets = [clock.offset * 1e3 for clock in fitted]
        drifts = [clock.drift * 1e6 for clock in fitted]
        spread = "offset ms %.1f/%.1f/%.1f drift ppm %.0f/%.0f/%.0f" % (
            min(offsets),
            median(offsets),
            max(offsets),
            min(drifts),
            median(drifts),
            max(drifts),
        )
        return f"{len(fitted)} fitted, {len(clocks) - len(fitted)} fell back; {spread}"


FALLBACK = Clock(nominal=0.0, offset=0.021, drift=537e-6, fitted=False)
"""The audit's constant model: its offset and drift stand in for a play whose own fit is not believable."""
OFFSET_BOUNDS = (0.0, 0.060)
DRIFT_BOUNDS = (400e-6, 700e-6)
MAX_RESIDUAL = 0.015
"""Seconds a window's lag may stray from the fitted line before the fit is not believable."""


@dataclass(frozen=True)
class Envelope:
    """A zero-mean unit-variance amplitude envelope at `RATE`."""

    samples: Float[np.ndarray, "t"]

    @classmethod
    def of(cls, signal: Float[np.ndarray, "t"], rate: float) -> "Envelope":
        """The rectified `signal` at `rate` Hz, smoothed below `CUTOFF` and brought to `RATE`."""
        rectified = np.abs(signal - signal.mean())
        smooth = sosfiltfilt(butter(FILTER_ORDER, CUTOFF, fs=rate, output="sos"), rectified)
        if rate != RATE:
            smooth = resample_poly(smooth, RATE, round(rate))
        return cls((smooth - smooth.mean()) / smooth.std())

    def span(self, start: int, stop: int) -> "Envelope":
        return Envelope(self.samples[start:stop])

    def correlation(self, reference: "Envelope") -> Float[np.ndarray, "l"]:
        """Pearson correlation of `reference` with this envelope at every position it fits, in samples."""
        length = len(reference.samples)
        centred = reference.samples - reference.samples.mean()
        cross = fftconvolve(self.samples, centred[::-1], mode="valid")
        total = np.concatenate([[0.0], np.cumsum(self.samples)])
        squares = np.concatenate([[0.0], np.cumsum(self.samples**2)])
        window_sum = total[length:] - total[:-length]
        spread = squares[length:] - squares[:-length] - window_sum**2 / length
        return cross / np.sqrt(np.maximum(spread, VARIANCE_FLOOR) * (centred**2).sum())


@dataclass(frozen=True)
class Alignment:
    """The played clip's envelope, against which a recording's audio channel is read for each play's `Clock`."""

    reference: Envelope
    window: int = WINDOW

    @classmethod
    def of(cls, wav: Path) -> "Alignment":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            rate, audio = wavfile.read(wav)
        return cls(Envelope.of(audio.astype(np.float64).mean(axis=1), rate))

    @property
    def seconds(self) -> float:
        return len(self.reference.samples) / RATE

    def locate(self, recorded: Envelope) -> list[float]:
        """Nominal starts (seconds) of the clip's plays found from the audio alone, for a run with no start trigger:
        the best matches that do not overlap, earliest first, each placed as the trigger would have been."""
        length = len(self.reference.samples)
        match = recorded.correlation(self.reference)
        peaks = []
        for _ in range(PLAYS_PER_RUN):
            peak = int(np.argmax(match))
            peaks.append(peak)
            match[max(peak - length, 0) : peak + length] = -np.inf
        middle = FALLBACK.offset + FALLBACK.drift * self.seconds / 2
        return [peak / RATE - middle for peak in sorted(peaks)]

    def lags(self, recorded: Envelope, nominal: float) -> tuple[Float[np.ndarray, "w"], Float[np.ndarray, "w"]]:
        """Each window's clip centre and best recorded lag (seconds) against a play nominally starting at `nominal`."""
        first = round(nominal * RATE)
        centres: list[float] = []
        lags: list[float] = []
        for begin in range(0, len(self.reference.samples) - self.window, self.window):
            segment = recorded.span(first + begin + LAG_LOW, first + begin + self.window + LAG_HIGH)
            match = segment.correlation(self.reference.span(begin, begin + self.window))
            centres.append((begin + self.window / 2) / RATE)
            lags.append((int(np.argmax(match)) + LAG_LOW) / RATE)
        return np.array(centres), np.array(lags)

    def clock(self, recorded: Envelope, nominal: float) -> Clock:
        """The play's fitted `Clock`, or `FALLBACK`'s constants if its recording is too short, the fit leaves the
        measured offset and drift ranges, or a window strays from the line."""
        first = round(nominal * RATE)
        base = first / RATE
        needed = first + len(self.reference.samples) + LAG_HIGH
        if first + LAG_LOW < 0 or needed > len(recorded.samples):
            return Clock.fallen(nominal, "the recording does not cover the play")
        centres, lags = self.lags(recorded, nominal)
        drift, offset = np.polyfit(centres, lags, 1)
        residual = float(np.abs(lags - (offset + drift * centres)).max())
        believable = (
            OFFSET_BOUNDS[0] <= offset <= OFFSET_BOUNDS[1]
            and DRIFT_BOUNDS[0] <= drift <= DRIFT_BOUNDS[1]
            and residual <= MAX_RESIDUAL
        )
        if not believable:
            return Clock.fallen(
                nominal,
                f"fit offset {offset * 1e3:.1f} ms drift {drift * 1e6:.0f} ppm residual {residual * 1e3:.1f} ms",
            )
        return Clock(base, float(offset), float(drift))

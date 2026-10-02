"""The holdout's word window and the recording constants every cut shares.

The holdout gives each word 1 s of 250 Hz MEG from its onset, already band-passed 0.1–125 Hz; `Window` is the
transform it goes through. Training windows are gathered from the continuous cache (`megspeech.data.windows`),
decimated by the same `DECIMATE` and centred the same way.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from scipy.signal import resample_poly

SFREQ = 250
WINDOW = SFREQ
"""1 s from onset — `LibriBrainCompetitionHoldout.WINDOW_SECONDS` at the recordings' 250 Hz."""

DECIMATE = 5
"""250 → 50 Hz: d'Ascoli 2025 decode at 50 Hz; `resample_poly`'s anti-alias filter keeps < 25 Hz."""

PICO = 1e12
"""Tesla → picotesla, so a float16 cache keeps the ~1e-13 T magnetometer signal out of subnormals."""

HALF = float(np.finfo(np.float16).max)
WORKERS = 6


@dataclass(frozen=True)
class Window:
    """The one transform every 1 s window goes through, training and holdout alike."""

    decimate: int = DECIMATE

    def __call__(self, raw: Float[np.ndarray, "n c t"]) -> Float[np.ndarray, "n c s"]:
        """Decimate, express in picotesla, and remove each channel's mean over the window."""
        down = resample_poly(raw.astype(np.float64) * PICO, 1, self.decimate, axis=-1)
        return (down - down.mean(axis=-1, keepdims=True)).astype(np.float32)

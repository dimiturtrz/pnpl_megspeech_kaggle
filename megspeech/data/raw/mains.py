"""The mains frequency a recording was made under, and the notch that removes it and its harmonics below LibriBrain's
125 Hz."""

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from megspeech.data.constants import SFREQ

BAND_EDGE = SFREQ / 2
"""LibriBrain's low-pass: the Nyquist of its 250 Hz, above which no harmonic is kept."""
POWER_LINE = "PowerLineFrequency"


@dataclass(frozen=True)
class Mains:
    """A recording's mains frequency in Hz."""

    frequency: float

    @classmethod
    def of(cls, sidecar: Path, fallback: float | None = None) -> "Mains":
        """The `PowerLineFrequency` of a BIDS `_meg.json`; `fallback` when it records none (`"n/a"`), else an error."""
        recorded = json.loads(sidecar.read_text()).get(POWER_LINE)
        if isinstance(recorded, int | float):
            return cls(float(recorded))
        if fallback is None:
            raise ValueError(f"{sidecar.name} records no {POWER_LINE}: {recorded!r}")
        return cls(fallback)

    @property
    def harmonics(self) -> list[float]:
        """The frequency and every multiple of it below `BAND_EDGE`."""
        return [self.frequency * order for order in range(1, int(BAND_EDGE / self.frequency) + 1)]

    def notch(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """`raw` with `harmonics` notched out, in place."""
        return raw.notch_filter(np.array(self.harmonics), verbose="error")

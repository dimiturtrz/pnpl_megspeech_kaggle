"""Artifact cleaning of a continuous 250 Hz recording before it is decimated into a cache.

`Cleaning.ICA` removes the heartbeat and eye-movement components an independent component analysis finds in each
recording on its own: the cardiac field and blinks are the largest non-neural sources in MEG, and a sensor-space
model has to learn to ignore them in every window.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import h5py
import mne
import numpy as np
from jaxtyping import Float

from megspeech.data.constants import SFREQ

LOG = logging.getLogger(__name__)

HIGHPASS = 1.0
"""Hz the ICA's training copy is high-passed at: MNE's guidance, since slow drifts break the components'
independence and swamp the fit, while the cleaned recording keeps its own 0.1 Hz band."""
VARIANCE = 0.99
"""Share of variance the ICA keeps in components: the heartbeat and blinks are high-variance, so they sit inside it;
SSS leaves the 306 channels a rank of ~70, which this keeps below rather than fitting null directions."""
DECIM = 5
"""Every fifth sample fits the ICA: 50 Hz still holds the cardiac and ocular band below 15 Hz, at a fifth of the
fit's cost."""
FRONTAL = ("MEG0511", "MEG0521", "MEG0531", "MEG0811", "MEG0911", "MEG0921", "MEG0941")
"""Frontal magnetometers over the eyes, the EOG proxy for a recording that has no EOG channel."""
ECG_Z = 3.0
"""z-score over components of their correlation with the synthetic ECG that marks one cardiac. MNE's default
cross-trial phase statistic fires on no component of an SSS'd run here (ses-10 Sherlock1: 1128 beats at 84 bpm
detected, best phase statistic 0.06), while one component correlates 0.44 against a field of ~0.26."""
SPAN = 2.0
"""Seconds per stretch the fit judges for rejection, MNE's default `tstep`."""
REJECT = 10.0
"""Multiple of a recording's median stretch swing past which the fit drops a stretch. A blink or heartbeat lifts the
swing a few-fold (99th percentile 3-4x the median over three Sherlock1 runs) and must stay in the fit to be found;
a sensor transient is orders beyond it (ses-3 Sherlock1: one second holds 98% of the first-difference energy, so
PCA gives it one component of 99.9% of the variance). MNE's fixed 4 pT magnetometer threshold drops 26-61% here."""
SEPARATOR = ","

Cleaner = Callable[[Float[np.ndarray, "c t"]], Float[np.ndarray, "c t"]]


@dataclass(frozen=True)
class Unchanged:
    """The recording as it is."""

    def __call__(self, meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "c t"]:
        return meg


UNCHANGED = Unchanged()


@dataclass(frozen=True)
class Ica:
    """ICA cleaning of a recording laid out as `names` of `types` (`mag`/`grad`), in tesla."""

    names: tuple[str, ...]
    types: tuple[str, ...]

    @classmethod
    def of(cls, meg: Path) -> Self:
        """The layout a serialised run's h5 attributes record."""
        with h5py.File(meg) as stored:
            names, types = (
                tuple(part.strip() for part in str(stored.attrs[key]).split(SEPARATOR))
                for key in ("channel_names", "channel_types")
            )
        return cls(names, types)

    @staticmethod
    def stretches(meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "c n s"]:
        """`meg` cut into its whole `SPAN`-long stretches, the remainder dropped."""
        span = round(SPAN * SFREQ)
        whole = meg.shape[1] // span * span
        return meg[:, :whole].reshape(len(meg), -1, span)

    def rejection(self, highpassed: Float[np.ndarray, "c t"]) -> dict[str, float]:
        """Per channel type, the peak-to-peak over a `SPAN` past which the fit drops that stretch: `REJECT` times the
        recording's median stretch's largest swing."""
        swing = np.ptp(Ica.stretches(highpassed), axis=2)
        kinds = np.array(self.types)
        return {kind: REJECT * float(np.median(swing[kinds == kind].max(axis=0))) for kind in set(self.types)}

    def _fitted(self, raw: mne.io.RawArray) -> mne.preprocessing.ICA:
        """The ICA of `raw`'s high-passed copy, its outlying stretches dropped."""
        highpassed = mne.filter.filter_data(raw.get_data(), SFREQ, HIGHPASS, None, copy=False, verbose="error")
        ica = mne.preprocessing.ICA(
            n_components=VARIANCE,
            method="infomax",
            fit_params={"extended": True},
            max_iter="auto",
            random_state=0,
            verbose="error",
        )
        reject = self.rejection(highpassed)
        return ica.fit(
            mne.io.RawArray(highpassed, raw.info, verbose="error"),
            decim=DECIM,
            reject=reject,
            tstep=SPAN,
            verbose="error",
        )

    @staticmethod
    def _energy(meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, " n"]:
        """Each `SPAN`-long stretch's first-difference energy."""
        return np.square(Ica.stretches(np.diff(meg, axis=1))).sum(axis=(0, 2), dtype=np.float64)

    def __call__(self, meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "c t"]:
        """`meg` without the components that track the synthetic ECG or the frontal magnetometers."""
        info = mne.create_info(list(self.names), SFREQ, list(self.types), verbose="error")
        raw = mne.io.RawArray(meg.astype(np.float64), info, verbose="error")
        ica = self._fitted(raw)
        cardiac, _ = ica.find_bads_ecg(raw, method="correlation", threshold=ECG_Z, verbose="error")
        ocular, _ = ica.find_bads_eog(raw, ch_name=list(FRONTAL), verbose="error")
        ica.exclude = sorted(set(cardiac) | set(ocular))
        cleaned = ica.apply(raw, verbose="error").get_data().astype(meg.dtype)
        del raw
        removed = float(np.median(Ica._energy(meg - cleaned) / Ica._energy(meg)))
        LOG.info(
            "ICA %d components, ECG %d, EOG %d excluded, median stretch lost %.3f of its first-difference energy",
            ica.n_components_,
            len(cardiac),
            len(ocular),
            removed,
        )
        return cleaned


class Cleaning(StrEnum):
    """What is removed from a recording before it is cached; `NONE` caches it as recorded."""

    NONE = ""
    ICA = "ica"

    def cleaner(self, layout: Callable[[], Path]) -> Cleaner:
        """This cleaning of a recording laid out as the serialised run `layout` names, read only if needed."""
        builders: dict[Cleaning, Callable[[], Cleaner]] = {
            Cleaning.NONE: Unchanged,
            Cleaning.ICA: lambda: Ica.of(layout()),
        }
        return builders[self]()

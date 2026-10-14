"""The recording constants every cut shares.

The recordings, training and Kaggle test alike, are 250 Hz MEG already band-passed 0.1–125 Hz, and a word's window is
1 s from its onset. Every signal is decimated by `DECIMATE` into the run cache (`cache.run_cache.RunCache`) and
every window gathered from it (`batching.windows.Windows`).
"""

import numpy as np

SFREQ = 250
WINDOW = SFREQ
"""1 s from onset — `LibriBrainCompetitionHoldout.WINDOW_SECONDS` at the recordings' 250 Hz."""

DECIMATE = 5
"""250 → 50 Hz: d'Ascoli 2025 decode at 50 Hz; `resample_poly`'s anti-alias filter keeps < 25 Hz."""

RATE = SFREQ // DECIMATE
"""50 Hz: the cached rate, and the rate a word onset in seconds is multiplied by to find its sample."""

TAIL = 1.0
"""Seconds a Kaggle test sentence epoch runs past its sentence's last word (`sentence_extra_end_s`), measured from the
word's end: the epochs' tails after their last onset are never under 1.022 s, though one in nine training sentences
ends on a vocabulary word whose tail from its onset would be exactly 1 s."""

QUIET = "error"
"""MNE log level for every library call: info-level chatter (filter design, projectors) is noise at corpus scale."""

PICO = 1e12
"""Tesla → picotesla, so a float16 cache keeps the ~1e-13 T magnetometer signal out of subnormals."""

HALF = float(np.finfo(np.float16).max)
WORKERS = 6
CHANNELS = 306

MISSING = [""]
"""The only cell an events table leaves empty for a missing value; `null` is a word."""

"""The compressed on-disk form of a rewritten run's MEG, and the one reader every consumer of a serialised run shares.

`data` is stored float16 after dividing each channel by its own standard deviation (kept as the float32 dataset
`scale`), gzip-compressed with the byte-shuffle filter, chunked along time. A run without `scale` is pnpl's own
float32 file and reads unchanged.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import pandas as pd
from jaxtyping import Float

LOG = logging.getLogger(__name__)

DATA = "data"
SCALE = "scale"
GZIP_LEVEL = 4
TIME_CHUNK = 2500
"""Samples per chunk along time: 10 s at 250 Hz, so a window read decompresses a handful of chunks."""
HALF_MAX = float(np.finfo(np.float16).max)
SERIALISED = "serialised"
EVENTS = "events"
"""`Run.events` reads a run's events from `events/` beside the MEG's own directory."""


@dataclass(frozen=True)
class Serialised:
    """One run's MEG file."""

    path: Path

    def write(self, meg: Float[np.ndarray, "c t"], attrs: Mapping[str, object]) -> None:
        """Store `meg` (tesla) as scaled float16 with `attrs` as attributes. A flat channel's scale is 1."""
        deviation = meg.std(axis=1)
        scale = np.where(deviation > 0, deviation, 1.0).astype(np.float32)
        scaled = np.clip(meg / scale[:, None], -HALF_MAX, HALF_MAX).astype(np.float16)
        with h5py.File(self.path, "w") as stored:
            stored.create_dataset(
                DATA,
                data=scaled,
                chunks=(scaled.shape[0], min(TIME_CHUNK, scaled.shape[1])),
                compression="gzip",
                compression_opts=GZIP_LEVEL,
                shuffle=True,
            )
            stored.create_dataset(SCALE, data=scale)
            stored.attrs.update(attrs)

    @staticmethod
    def read(stored: h5py.File) -> Float[np.ndarray, "c t"]:
        """The MEG of an open run file as float32 tesla, whichever form it was written in."""
        meg = stored[DATA][:]
        if SCALE not in stored:
            return meg
        return meg.astype(np.float32) * stored[SCALE][:][:, None]


class Convertible[T](Protocol):
    """One downloaded recording that becomes a run on a `T` sensor layout."""

    @property
    def name(self) -> str:
        """The run name `Run` parses."""
        ...

    def written(self, target: T) -> tuple[Float[np.ndarray, "c t"], Mapping[str, object]]:
        """The recording's MEG on `target` in tesla and the h5 attributes describing it."""
        ...

    def events(self) -> pd.DataFrame:
        """The recording's events in pnpl's columns."""
        ...


@dataclass(frozen=True)
class Corpus[T]:
    """A study's downloaded recordings at `source` and the LibriBrain-shaped runs they become under `target`."""

    source: Path
    target: Path

    def meg(self, recording: Convertible[T]) -> Path:
        return self.target / SERIALISED / f"{recording.name}_meg.h5"

    def events(self, recording: Convertible[T]) -> Path:
        return self.target / EVENTS / f"{recording.name}_events.tsv"

    def convert(self, recording: Convertible[T], layout: T) -> Path:
        """Write `recording` as a run unless already written; its events land last, so `Run.discover` never finds a
        run whose MEG is partial."""
        meg, events = self.meg(recording), self.events(recording)
        if events.exists():
            return meg
        meg.parent.mkdir(parents=True, exist_ok=True)
        events.parent.mkdir(parents=True, exist_ok=True)
        signal, attrs = recording.written(layout)
        Serialised(meg).write(signal, attrs)
        recording.events().to_csv(events, sep="	", index=False)
        LOG.info("-> %s", recording.name)
        return meg

"""The holdout as continuous 50 Hz segments, cached and gathered exactly as the training runs are.

Every holdout epoch (`archive.Archive`) is a segment of continuous MEG, so each goes through `RunCache.decimated` on
its own — the transform a training run takes whole — and an archive's segments are cached end to end as one run of a
run cache. A holdout word's window is then gathered by `batching.windows.Windows` from its onset sample, as a
training word's is. Only an isolated word, whose segment IS its window, still carries the decimation filter's edges:
the holdout holds nothing either side of it.

Rows follow the competition's canonical order: subjects ascending, the sentence archive before the word archive,
epochs in stored order, a sentence's valid words in stored order.

    uv run python -m megspeech.data.cache.holdout     # cache the Deep holdout and report its windows
"""

import argparse
import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from pnpl.competition.holdout import DEEP_SUBJECTS, HOLDOUT_SUBDIR, TRACKS, LibriBrainCompetitionHoldout

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.archive import Archive, ArchiveKind
from megspeech.data.cache.recording import DeviceRecording
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import DECIMATE

LOG = logging.getLogger(__name__)

HOLDOUT_DIR = "holdout"
"""The holdout's cache, a directory inside the training cache that `DeviceRecording.load` never globs into."""


class Track(StrEnum):
    """A competition track: the holdout subjects it scores, in canonical order."""

    DEEP = "deep"
    BROAD = "broad"
    """Subjects 1-39; 33-39 have no training data, so a model meets them zero-shot."""

    @property
    def subjects(self) -> tuple[int, ...]:
        return TRACKS[self.value]

    def holdout(self, raw: Path) -> LibriBrainCompetitionHoldout:
        """The track's rows as the competition indexes them, from the archives under `raw`."""
        return LibriBrainCompetitionHoldout(self.value, data_path=raw, download=False)


@dataclass(frozen=True)
class HoldoutCache:
    """The holdout archives of `subjects`, cached into `cache` and gathered as one `Windows` in canonical order."""

    raw: Path
    cache: RunCache
    subjects: tuple[int, ...] = DEEP_SUBJECTS

    @classmethod
    def of(cls, root: DataRoot, decimate: int = DECIMATE, track: Track = Track.DEEP) -> "HoldoutCache":
        """`track`'s holdout cached inside the training cache at `decimate`."""
        trained = RunCache.under(root.processed(), decimate)
        return cls(root.raw() / HOLDOUT_SUBDIR, RunCache(trained.cache / HOLDOUT_DIR, decimate), track.subjects)

    def archives(self) -> list[Archive]:
        return [
            Archive(self.raw / f"subj{subject:02d}_{kind}.npz") for subject in self.subjects for kind in ArchiveKind
        ]

    def build(self) -> pd.DataFrame:
        """Cache every archive, then write the word table of all of them."""
        self.cache.ready()
        archives = self.archives()
        for archive in archives:
            LOG.info("%s", self.cache.write(archive.name, partial(archive.signal, self.cache)).name)
        return self.cache.publish(pd.concat([archive.words(self.cache) for archive in archives], ignore_index=True))

    def windows(self, device: torch.device | str, length: int, *, standardised: bool = False) -> Windows:
        """Every holdout row's `length`-sample window, in canonical order, from the built cache, each archive
        `standardised` on its own if asked."""
        names = [archive.name for archive in self.archives()]
        return DeviceRecording.load(self.cache, device, names, standardised=standardised).windows(length)


def main() -> None:
    """Cache a track's holdout, then report its windows' shape, count against the holdout's rows, finite share, and
    the share of samples zeroed past their epoch's end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=float, default=1.0, help="window length in seconds")
    parser.add_argument("--decimate", type=int, default=DECIMATE)
    parser.add_argument("--track", type=Track, choices=list(Track), default=Track.DEEP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    cache = HoldoutCache.of(root, args.decimate, args.track)
    cache.build()
    rows = len(args.track.holdout(root.raw()))
    windows = cache.windows(args.device, round(args.window * cache.cache.rate))
    x = windows(torch.arange(len(windows), device=windows.device))
    finite = float(torch.isfinite(x).float().mean())
    zeroed = float((x == 0).all(dim=1).float().mean())
    shape = tuple(x.shape[1:])
    LOG.info(
        "%d windows of %s for %d holdout rows, finite %.6f, zeroed %.6f", len(windows), shape, rows, finite, zeroed
    )


if __name__ == "__main__":
    main()

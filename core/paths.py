"""Machine-local data locations, adapted to the OS the process is actually running on.

One data root per machine, recorded in the gitignored `paths.yaml`. `MEGSPEECH_DATA` overrides it, so a
runner with no dataset can point at an empty directory and the data-gated tests skip instead of erroring.

The root is per MODALITY (`D:/data/neural`), shared with every project reading that modality. Each dataset has ONE
slug, the same under every kind and as its adapter's module name (`megspeech.data.raw.sources.<dataset>`):

    <data>/raw/<dataset>/                  as downloaded, never written to
    <data>/processed/<dataset>/            derived data; the competition's (`pnpl_libribrain`) holds
                                           features/, viz/, submissions/, studies/, dascoli_ref/ beside its caches
    <data>/models/<dataset>/               our checkpoints + their run records
    <data>/models/foreign/<name>/          third-party weights (MEG-XL, ...), ours or another project's

A kind resolves to `<data>/<kind>` unless paths.yaml names it as a top-level key, which redirects that
kind alone. Every configured path goes through `NativePath.resolve`, so a path written on one OS either
means something here or raises.
"""

import os
from pathlib import Path

import yaml

from core.native_path import NativePath

ENV_OVERRIDE = "MEGSPEECH_DATA"
"""Environment variable that replaces paths.yaml entirely. Named in the refusal `NativePath` raises."""

CONFIG = Path(__file__).resolve().parent.parent / "paths.yaml"
"""The machine-local config at the repo root (gitignored; `paths.example.yaml` is its template)."""

COMPETITION = "pnpl_libribrain"
"""Directory name the competition's data takes under `raw/`, `processed/` and `models/`."""

RAW = "raw"
"""The as-downloaded datasets. Never written to."""

PROCESSED = "processed"
"""Derived DATA — epochs, caches, embeddings, prediction caches. Not checkpoints: those are `MODELS`."""

MODELS = "models"
"""Trained weights and the record of the run that produced them. A sibling of `processed/`: a checkpoint
is not data, nothing trains on it."""

FOREIGN = "foreign"
"""Under `models/`: weights someone else trained, kept apart from ours so a roster never mixes them."""


class DataRoot:
    """The data root. Datasets sit under `raw/` untouched; derived artefacts and weights are created on demand."""

    def __init__(self, root: Path, kinds: dict[str, str] | None = None) -> None:
        self._root = root
        self._kinds = kinds or {}

    @classmethod
    def from_config(cls, config: Path = CONFIG) -> "DataRoot":
        """Resolve the root: the `MEGSPEECH_DATA` env var if set, else the `data:` entry of paths.yaml.

        The env override short-circuits the READ, not just the `data:` entry: a runner that sets it has no
        paths.yaml to open. Any other top-level key is a per-kind override.
        """
        override = os.environ.get(ENV_OVERRIDE)
        if override:
            return cls(Path(NativePath.resolve(override, ENV_OVERRIDE)))
        entries: dict[str, str] = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        return cls(Path(NativePath.resolve(str(entries["data"]), ENV_OVERRIDE)), entries)

    def kind(self, name: str) -> Path:
        """The root for one kind of data — `<data>/<name>`, or the path paths.yaml gave that name."""
        stated = self._kinds.get(name)
        return Path(NativePath.resolve(stated, ENV_OVERRIDE)) if stated else self._root / name

    def raw(self, dataset: str = COMPETITION) -> Path:
        """Where an as-downloaded dataset lives. Never created here — a download is read-only."""
        return self.kind(RAW) / dataset

    def processed(self, dataset: str = COMPETITION) -> Path:
        """Where derived artefacts for a dataset go. Created on demand."""
        return self._created(self.kind(PROCESSED) / dataset)

    def models(self, dataset: str = COMPETITION) -> Path:
        """Where a dataset's trained checkpoints and their run records go. Created on demand."""
        return self._created(self.kind(MODELS) / dataset)

    def foreign(self, name: str) -> Path:
        """Where third-party weights called `name` live. Created on demand, so a fetch can write into it."""
        return self._created(self.kind(MODELS) / FOREIGN / name)

    @staticmethod
    def _created(destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        return destination

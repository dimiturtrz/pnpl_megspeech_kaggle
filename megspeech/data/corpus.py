"""The on-device train set and validation domains a training run reads, cut from the run cache."""

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from core.paths import DataRoot
from megspeech.data.batching.split import TRAIN, Features, Split
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED, Geometry
from megspeech.data.cache.kaggle import KaggleCache, Track
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.heard import Heard
from megspeech.data.raw.runs import Run

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Corpus:
    """The on-device train set, the validation domains, and every word sequence training has seen."""

    train: Split
    validation: dict[str, Split]
    seen: frozenset[tuple[int, ...]]
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Each train word's `run`, `position` in that run's word table (`RunCache.words`), `onset` and
    `sentence`, in train-split order."""
    geometry: Geometry = AS_RECORDED
    """The sensor geometry every split was loaded in, which the Kaggle test must load in too."""

    @classmethod
    def load(cls, data: DataConfig, root: DataRoot, device: torch.device) -> "Corpus":
        cache = data.run_cache(root)
        runs = Run.everywhere(root)
        trained, held = data.trained(runs), data.held(runs)
        reading = [r.name for r in trained] + [r.name for group in held.values() for r in group]
        names = list(dict.fromkeys(reading))
        geometry = data.geometry(root, device)
        source = DeviceRecording.load(cache, device, names, geometry=geometry)
        table = source.words.assign(position=source.words.groupby("run").cumcount())
        words = data.carried(table)
        weights = {r.name: data.weight(r) for r in trained}
        chosen = words[words["run"].isin(weights)]
        weight = chosen["run"].map(weights).to_numpy(dtype=np.float64)
        train = cls.split(source, chosen, data, weight).recurring(data.repeat_power)
        positions = chosen[["run", "position", "onset", "sentence"]].reset_index(drop=True)
        validation = cls.domains(source, words, held, data)
        features = Features.under(root, data.features)
        train = features.split(TRAIN, train)
        validation = {name: features.split(name, split) for name, split in validation.items()}
        heard = Heard.of(cache, runs, data)
        LOG.info("cache %.2f GB on %s", source.gigabytes, device)
        LOG.info("train %d words (%d labelled) / %d spans", len(train.y), len(train.labelled()), len(train.index))
        muted = train.muted
        LOG.info(
            "context targets masked on %d of %d train words", 0 if muted is None else int(muted.sum()), len(train.y)
        )
        LOG.info("validation %s", {task: len(split.labelled()) for task, split in validation.items()})
        if len(positions) != len(train.y):
            raise ValueError(f"{len(positions)} train positions against {len(train.y)} train words")
        return cls(train, validation, Heard.sequences(heard, data.grouping), positions, geometry)

    @staticmethod
    def kaggle(
        root: DataRoot, data: DataConfig, track: Track, device: torch.device | str, geometry: Geometry = AS_RECORDED
    ) -> Windows:
        """`track`'s Kaggle test windows as `data` reads them, its cache built first if missing, every archive loaded
        in `geometry`."""
        cache = KaggleCache.of(root, data.decimate, track, data.cleaning)
        cache.build()
        windows = data.reading(cache.windows(device, data.samples, geometry=geometry))
        unread = windows.unread()
        if len(unread):
            raise ValueError(
                f"{len(unread)} of {len(windows)} Kaggle test windows start past their epoch's end: "
                f"delay {data.delay} s (+ window {data.window} s) outruns what the test holds per word"
            )
        return windows

    @staticmethod
    def split(source: DeviceRecording, words: pd.DataFrame, data: DataConfig, weight: Float[np.ndarray, " n"]) -> Split:
        """`words` (rows of `source.words`) as a split whose `data.window`-long windows are gathered from `source`'s
        device signal, each stopped where `data.grouping` bounds it."""
        windows = data.reading(
            Windows(
                source.meg,
                source.samples(words["at"]),
                source.samples(data.grouping.bound(words)),
                begin=source.samples(data.grouping.opening(words)),
                following=source.samples(data.grouping.following(words)),
            )
        )
        labels = (words["target"].to_numpy(), words["token"].to_numpy())
        held = Split.of(windows, labels, data.grouping.spans(words), weight, Run.subjects(words["run"]))
        return replace(
            held,
            source=torch.tensor(data.sources(words["run"]), device=windows.device),
            read=torch.tensor(data.reads(words["run"]), device=windows.device),
            muted=torch.tensor(data.muted(words["run"]), device=windows.device),
        )

    @classmethod
    def domains(
        cls, source: DeviceRecording, words: pd.DataFrame, held: dict[str, list[Run]], data: DataConfig
    ) -> dict[str, Split]:
        """Each validation domain's `words` (`DataConfig.held`), one split per domain so each is scored alone."""
        chosen = {name: words[words["run"].isin([r.name for r in runs])] for name, runs in held.items()}
        return {name: cls.split(source, rows, data, np.ones(len(rows))) for name, rows in chosen.items()}

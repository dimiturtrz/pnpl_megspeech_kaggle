"""Labelled runs read as a trained model reads them: its data config's cache, geometry, grouping and features."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import Features, Split
from megspeech.data.cache.alignment import AS_RECORDED, Geometry
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Column, Partition, Run
from megspeech.evaluation.text import RULER, Text
from megspeech.model.input.subjects import Reading


@dataclass(frozen=True)
class Reader:
    """Runs on `device` as a model trained under `data` reads them, each word presented as `reading`'s subject; the
    scored domains are `partition`'s: val's or the local test's."""

    root: DataRoot
    data: DataConfig
    device: torch.device
    reading: Reading = Reading.OWN
    geometry: Geometry = AS_RECORDED
    partition: Partition = Partition.VALIDATION

    def load(self, runs: list[Run]) -> tuple[DeviceRecording, pd.DataFrame]:
        """`runs` on the device, and their words keyed by `Text` and cut as the model's grouping cuts them."""
        cache = self.data.run_cache(self.root)
        names = [run.name for run in runs]
        source = DeviceRecording.load(cache, self.device, names, geometry=self.geometry)
        return source, self.data.carried(Text.of(source.words).words)

    def spoken(self, runs: list[Run]) -> tuple[pd.DataFrame, pd.DataFrame]:  # devtools-ignore: test-mirror
        """`runs`' every word keyed by `Text` (no MEG read), and those of them the model's grouping carries."""
        full = Text.of(self.data.run_cache(self.root).table([run.name for run in runs])).words
        return full, self.data.carried(full)

    @property
    def features(self) -> Features:
        """What the model reads in place of MEG windows, if it trained on saved features."""
        return Features.under(self.root, self.data.features)

    def read(self, name: str, split: Split) -> Split:
        """`split` as the model reads it: `name`'s saved features if it trained on them, each word presented as
        `reading`'s subject."""
        split = self.features.split(name, split)
        return replace(split, subject=self.reading.subjects(split.subject))

    def splits(self, runs: list[Run]) -> dict[Run, tuple[pd.DataFrame, Split]]:
        """Each of `runs`, loaded together: its words and its split as the model reads it."""
        source, words = self.load(runs)
        rows = {run: words[words[Column.RUN] == run.name] for run in runs}
        return {
            run: (part, self.read(run.name, Corpus.split(source, part, self.data, np.ones(len(part)))))
            for run, part in rows.items()
        }

    def domains(self, held: dict[str, list[Run]]) -> tuple[dict[str, Split], pd.DataFrame]:
        """Subject 0's validation domains as the model reads them, and the ruler domain's words in its split's order."""
        source, words = self.load([run for group in held.values() for run in group])
        ruler = words[words[Column.RUN].isin([run.name for run in held.get(RULER, [])])]
        domains = Corpus.domains(source, words, held, self.data)
        return {name: self.read(name, split) for name, split in domains.items()}, ruler

    def ruled(self, runs: list[Run]) -> dict[str, list[Run]]:
        """The domains of `runs` scored: val's or the local test's under the model's own `val`."""
        return self.data.ruled(runs, self.partition)

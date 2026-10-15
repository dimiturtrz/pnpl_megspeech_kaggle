"""Export a stratified sample set of the local test and each model's outputs on it, for the viewer to browse."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.raw.runs import DEEP, Column, Partition, Run
from megspeech.data.sources import Sources
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.viz.media import Stimulus
from megspeech.evaluation.viz.outputs import Reader, Reading
from megspeech.evaluation.viz.records import SAMPLES, Builder, SampleSet, Window
from megspeech.evaluation.viz.sample import STRATA, Picker, Profiles, Sentences
from megspeech.evaluation.viz.sensors import Helmet, Layout

VIZ = "viz"
MANIFEST = "manifest.json"
MODELS = "models"
LOCAL_TEST = (Partition.VALIDATION, Partition.TEST)
"""pnpl's validation and test partitions of subject 0: the local test pool the samples are drawn from."""


@dataclass(frozen=True)
class ExportConfig:
    models: tuple[str, ...]
    n: int = 1000
    """Sentences drawn; every labelled word in them is a sample."""
    seed: int = 0
    name: str = "default"


class Sampling(Protocol):
    def draw(self, root: DataRoot, config: ExportConfig, runs: list[Run]) -> SampleSet: ...


@dataclass(frozen=True)
class Sampler:
    """Draws the sample set from the first model's reading of the local test: its labelled words, its MEG."""

    device: str = Device.CPU

    def draw(  # devtools-ignore: test-mirror
        self, root: DataRoot, config: ExportConfig, runs: list[Run]
    ) -> SampleSet:
        loaded = Loader.of(Group(root, Scoring(), torch.device(self.device)), [config.models[0]])
        source, words = loaded.load(runs)
        pool = words[words[Column.TARGET] != IGNORE].reset_index(drop=True)
        books = tuple(sorted(Run.tasks(pool[Column.RUN]).unique()))
        sources = Run.tasks(pool[Column.RUN]).map(Sources.stratum)
        picked = Picker(seed=config.seed).whole(Profiles.of(pool, books), Sentences.keys(pool), sources, config.n)
        bounds = dict(zip(source.names, zip(source.offsets[:-1], source.offsets[1:], strict=True), strict=True))
        rows = pool.iloc[picked].reset_index(drop=True)
        rows = rows.assign(at=rows["at"] - rows[Column.RUN].map(lambda run: bounds[run][0]))
        signals = {
            run: source.meg[:, low:high].float().cpu().numpy()
            for run, (low, high) in bounds.items()
            if run in set(rows[Column.RUN])
        }
        stimuli = {run.name: Stimulus.of(run) for run in runs if run.name in signals}
        length = round(loaded.data.window * loaded.data.rate)
        window = Window(loaded.data.rate, length, pad=length // 2)
        sentences = Sentences(words)
        return Builder(window, sentences).build(rows, sentences.lengths(pool), signals, stimuli)


@dataclass(frozen=True)
class Exporter:
    """Writes `<data>/viz/<name>/`: the manifest, the samples and their media, each model's outputs."""

    root: DataRoot | None = None
    sampler: Sampling = field(default_factory=Sampler)
    reader: Reading = field(default_factory=Reader)
    helmet: Layout = field(default_factory=Helmet)

    @staticmethod
    def runs(root: DataRoot) -> list[Run]:
        """Subject 0's local test runs."""
        return [r for r in Run.discover(root.raw()) if r.subject == DEEP and r.partition in LOCAL_TEST]

    def run(self, config: ExportConfig) -> Path:
        root = self.root or DataRoot.from_config()
        out = root.kind(VIZ) / config.name
        runs = self.runs(root)
        samples = self.sampler.draw(root, config, runs)
        samples.write(out)
        outputs = [self.reader.read(root, name, runs, samples.keys) for name in config.models]
        for output in outputs:
            output.write(out / MODELS)
        manifest = {
            "name": config.name,
            "seed": config.seed,
            "strata": list(STRATA),
            "models": [output.manifest() for output in outputs],
            "samples": SAMPLES,
            "sensors": self.helmet.describe(root),
        }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return out

"""What a training run reads: which runs train and at what weight, cut from the continuous cache with validation."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.continuous import DIRECTORY, Continuous
from megspeech.data.epochs import RATE
from megspeech.data.grouping import Grouping
from megspeech.data.runs import Partition, Run
from megspeech.data.split import Split
from megspeech.data.vocabulary import ABSENT, IGNORE, PRIMARY
from megspeech.data.windows import Windows

LOG = logging.getLogger(__name__)

METRICS = "metrics.json"
"""The record a finished run leaves in its directory, its config under `config`."""

READ = frozenset({"TIMIT", "MOCHATIMIT"})
"""Tasks of read sentences, the holdout's domain; the rest is narrative (Sherlock, TheMoth)."""


@dataclass(frozen=True)
class DataConfig:
    tasks: tuple[str, ...] = ()
    """Training restricted to these tasks' runs; empty trains on every task."""
    read_weight: float = 1.0
    """Sampling weight of read-sentence runs (TIMIT, MOCHA-TIMIT) — the holdout's domain — against narrative ones."""
    repeat_power: float = 0.0
    """A span heard `k` times in training is drawn with weight `k ** -repeat_power`; 1 draws each sentence once."""
    grouping: Grouping = Grouping.SENTENCE
    """`sentence`: every word of each PNPL sentence; `gap`: vocabulary words only, spans by onset gap."""
    refit: bool = False
    """Train on the validation runs too, for a final fit of an already selected recipe; validation then scores
    words training heard, so it can no longer select."""
    window: float = 1.0
    """Seconds of MEG from each word's onset; a model is scored and submitted at the window it trained on."""
    standardised: bool = False
    """Each recording — training or validation run, holdout archive — robust-scaled per channel by its own median and
    IQR before the model's scaler. Across the 162 runs a channel's IQR over its median run spans 0.78-1.28 (p5-p95);
    the holdout sentence archive's sits at 0.90."""

    @classmethod
    def saved(cls, directory: Path) -> "DataConfig":
        """The data config a run in `directory` recorded; a record without one, or a flat one, falls to defaults."""
        config = json.loads((directory / METRICS).read_text())["config"]
        return Tree(cls).load(config.get("data", {}))

    @classmethod
    def shared(cls, directories: Sequence[Path], grouping: Grouping) -> "DataConfig":
        """The holdout reading — window and standardisation — every run in `directories` trained on, spans cut by
        `grouping`; members that read MEG differently cannot share one set of holdout windows."""
        readings = {(saved.window, saved.standardised) for saved in map(cls.saved, directories)}
        if len(readings) != 1:
            raise ValueError(f"runs trained at (window, standardised) {sorted(readings)}; an ensemble needs one")
        window, standardised = readings.pop()
        return cls(grouping=grouping, window=window, standardised=standardised)

    @property
    def samples(self) -> int:
        """The window in samples at the cache's `RATE`."""
        return round(self.window * RATE)

    def trains_on(self, run: Run) -> bool:
        trained = (Partition.TRAIN, Partition.VALIDATION) if self.refit else (Partition.TRAIN,)
        return run.partition in trained and (not self.tasks or run.key[0] in self.tasks)


@dataclass(frozen=True)
class Corpus:
    """The on-device train set, the validation domains, and every word sequence training has seen."""

    train: Split
    validation: dict[str, Split]
    seen: frozenset[tuple[int, ...]]

    @classmethod
    def load(cls, data: DataConfig, root: DataRoot, device: torch.device) -> "Corpus":
        cache = root.processed() / DIRECTORY
        runs = Run.discover(root.raw())
        trained = [r for r in runs if data.trains_on(r)]
        held = [r for r in runs if r.partition is Partition.VALIDATION]
        source = Continuous.load(cache, device, [r.name for r in trained + held], standardised=data.standardised)
        words = cls.words(source.words, data.grouping)
        weights = {r.name: data.read_weight if r.key[0] in READ else 1.0 for r in trained}
        chosen = words[words["run"].isin(weights)]
        weight = chosen["run"].map(weights).to_numpy(dtype=np.float64)
        train = cls.split(source, chosen, data, weight).recurring(data.repeat_power)
        validation = cls.domains(source, words, held, data)
        heard = cls.heard(cache, runs, data.grouping)
        LOG.info("cache %.2f GB on %s", source.gigabytes, device)
        LOG.info("train %d words (%d labelled) / %d spans", len(train.y), len(train.labelled()), len(train.index))
        LOG.info("validation %s", {task: len(split.labelled()) for task, split in validation.items()})
        return cls(train, validation, cls.sequences(heard, data.grouping))

    @staticmethod
    def words(table: pd.DataFrame, grouping: Grouping) -> pd.DataFrame:
        """The words of `table` that `grouping` carries and whose window fits in their run, with their `target`
        and `token` (the lower-cased word, hashed)."""
        labels = PRIMARY.labels(table["word"])
        target = np.where(labels == ABSENT, IGNORE, labels)
        token = pd.util.hash_array(table["word"].astype(str).str.lower().to_numpy()).view(np.int64)
        return table.assign(target=target, token=token)[grouping.keep(target) & Continuous.inside(table)]

    @staticmethod
    def split(source: Continuous, words: pd.DataFrame, data: DataConfig, weight: Float[np.ndarray, " n"]) -> Split:
        """`words` (rows of `source.words`) as a split whose `data.window`-long windows are gathered from `source`'s
        device signal, each stopped where `data.grouping` bounds it."""
        windows = Windows(
            source.meg, source.samples(words["at"]), source.samples(data.grouping.bound(words)), data.samples
        )
        labels = (words["target"].to_numpy(), words["token"].to_numpy())
        return Split.of(windows, labels, data.grouping.spans(words), weight)

    @classmethod
    def domains(
        cls, source: Continuous, words: pd.DataFrame, held: Sequence[Run], data: DataConfig
    ) -> dict[str, Split]:
        """The official validation runs' `words`, one split per task so each domain is scored alone."""
        tasks = {task: [r.name for r in held if r.key[0] == task] for task in sorted({r.key[0] for r in held})}
        chosen = {task: words[words["run"].isin(names)] for task, names in tasks.items()}
        return {task: cls.split(source, rows, data, np.ones(len(rows))) for task, rows in chosen.items()}

    @classmethod
    def heard(cls, cache: Path, runs: Sequence[Run], grouping: Grouping) -> pd.DataFrame:
        """Every training-partition word `grouping` carries, read from the word table without any MEG."""
        names = [r.name for r in runs if r.partition is Partition.TRAIN]
        return cls.words(Continuous.table(cache, names), grouping)

    @staticmethod
    def sequences(heard: pd.DataFrame, grouping: Grouping) -> frozenset[tuple[int, ...]]:
        """The text of every span of `heard` as its token sequence, spans cut run by run."""
        runs = heard.groupby("run", sort=False)
        return Split.seen((run["token"].to_numpy(), grouping.spans(run)) for _, run in runs)

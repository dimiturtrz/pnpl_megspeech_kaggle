"""What a training run reads: which runs train and at what weight, gathered onto the device with validation."""

import logging
from dataclasses import dataclass

import torch

from core.paths import DataRoot
from megspeech.data.epochs import Epocher
from megspeech.data.runs import Partition, Run
from megspeech.data.split import Split

LOG = logging.getLogger(__name__)

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

    def trains_on(self, run: Run) -> bool:
        return run.partition is Partition.TRAIN and (not self.tasks or run.key[0] in self.tasks)


@dataclass(frozen=True)
class Corpus:
    """The on-device train set, the validation domains, and every word sequence training has seen."""

    train: Split
    validation: dict[str, Split]
    seen: frozenset[tuple[int, ...]]

    @classmethod
    def load(cls, data: DataConfig, root: DataRoot, device: torch.device) -> "Corpus":
        epocher = Epocher(root.processed() / "epochs")
        runs = Run.discover(root.raw())
        weighted = (
            (epocher.cached(r), data.read_weight if r.key[0] in READ else 1.0) for r in runs if data.trains_on(r)
        )
        train = Split.gather(weighted, device).recurring(data.repeat_power)
        validation = Split.validation(epocher, runs, device)
        seen = Split.seen(epocher.labelled(r) for r in runs if r.partition is Partition.TRAIN)
        LOG.info("train %d words / %d spans", len(train.y), len(train.index))
        LOG.info("validation %s", {task: len(split.y) for task, split in validation.items()})
        return cls(train, validation, seen)

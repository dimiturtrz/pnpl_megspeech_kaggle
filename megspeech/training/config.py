"""Everything a training run is made of, and the record a finished run leaves of it."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from core.tree import Tree
from megspeech.data.corpus import DataConfig
from megspeech.evaluation.report import MIN_NOVEL, Epoch
from megspeech.model.onsets import CONTEXT, ContextConfig
from megspeech.training.augment import Augment

METRICS = "metrics.json"


@dataclass(frozen=True)
class OptimConfig:
    epochs: int = 40
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    clip: float = math.inf
    seed: int = 0
    compiled: bool = True
    """Compile the model's forward and backward into CUDA graphs; batches have fixed shapes, so they replay."""


@dataclass(frozen=True)
class SelectionConfig:
    min_novel: int = MIN_NOVEL
    """Unseen-span words a domain needs to count toward selection: two per class of the 50 on average."""


@dataclass(frozen=True)
class RunConfig:
    """Everything a training run is made of; `core.tree.Tree` turns it into the command line."""

    data: DataConfig = DataConfig()
    context: ContextConfig = CONTEXT
    augment: Augment = Augment()
    optim: OptimConfig = OptimConfig()
    selection: SelectionConfig = SelectionConfig()

    def selected(self, history: list[Epoch]) -> Epoch:
        """The epoch whose weights the run keeps: the best on validation, or the last when validation was trained on."""
        return history[-1] if self.data.refit else max(history, key=lambda row: row.macro.bacc10)

    @classmethod
    def saved(cls, directory: Path) -> "RunConfig":
        """The config a run in `directory` recorded, flat records from before the tree included."""
        return Tree(cls).load(json.loads((directory / METRICS).read_text())["config"])

"""What scoring a model produces: one score per adjustment, one report per domain, one record per validation pass."""

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from core.tree import Tree
from megspeech.data.batching.split import BATCH, EVAL_SPANS
from megspeech.data.config import METRICS
from megspeech.evaluation.metric.ruler import Coverage, Interval

ADJUSTMENTS = (0.0, 0.5, 1.0)
"""Logit-adjustment strengths tried on validation: none, half, and the full Bayes correction."""

UNSELECTED = 0.5
"""The adjustment when no validation score counts (a refit trained on val): 239 of 243 val-selected Deep models
chose it (2026-10-14)."""

MIN_NOVEL = 100
"""Unseen-span words a domain needs to count toward selection: two per class of the 50 on average."""


@dataclass(frozen=True)
class SelectionConfig:
    min_novel: int = MIN_NOVEL
    """Unseen-span words a domain needs to count toward selection: two per class of the 50 on average."""

    @classmethod
    def saved(cls, directory: Path) -> "SelectionConfig":
        """The selection config a run in `directory` recorded; a record without one falls to defaults."""
        config = json.loads((directory / METRICS).read_text())["config"]
        return Tree(cls).load(config.get("selection", {}))

    @classmethod
    def shared(cls, directories: Sequence[Path]) -> "SelectionConfig":
        """The selection every run in `directories` was scored by: an ensemble counts a domain as its members did."""
        chosen = {cls.saved(directory) for directory in directories}
        if len(chosen) != 1:
            raise ValueError(f"runs selected by {sorted(chosen, key=repr)}; an ensemble needs one")
        return chosen.pop()


@dataclass(frozen=True)
class Forward:
    """How many spans one evaluation forward reads, beside a training step of `batch` spans."""

    batch: int = BATCH

    def spans(self) -> int:
        """As many times `batch` as `EVAL_SPANS` is `BATCH`, since a forward without gradients holds none of
        training's activations; a small batch for a heavy encoder shrinks both."""
        return max(1, EVAL_SPANS * self.batch // BATCH)

    @classmethod
    def fewest(cls, directories: Sequence[Path]) -> "Forward":
        """The smallest training batch the runs in `directories` saved; a record without one trained at `BATCH`."""
        configs = (json.loads((directory / METRICS).read_text())["config"] for directory in directories)
        return cls(min(config.get("optim", {}).get("batch", BATCH) for config in configs))


class Head(StrEnum):
    CONTEXT = "context"
    WORD = "word"


@dataclass(frozen=True)
class Score:
    tau: float
    bacc10: float
    by_tau: dict[float, float]
    n: int

    @classmethod
    def best(cls, by_tau: dict[float, float], n: int) -> "Score":
        counted = {tau: value for tau, value in by_tau.items() if not math.isnan(value)}
        tau = max(counted, key=counted.__getitem__, default=UNSELECTED)
        return cls(tau, by_tau[tau], by_tau, n)

    @classmethod
    def mean(cls, scores: Iterable["Score"]) -> "Score":
        """The macro of `scores`: their unweighted mean at each adjustment, the best adjustment chosen once for all."""
        chosen = list(scores)
        by_tau = {tau: cls.average(score.by_tau[tau] for score in chosen) for tau in ADJUSTMENTS}
        return cls.best(by_tau, sum(score.n for score in chosen))

    @staticmethod
    def average(values: Iterable[float]) -> float:
        """The mean of `values`, NaN when there are none: a macro over no counted domain scores nothing."""
        chosen = list(values)
        return float(np.mean(chosen)) if chosen else math.nan


@dataclass(frozen=True)
class Report:
    """One validation domain: both heads, the context head on Kaggle-test-like and on unseen spans, and — on the unseen
    spans, by logit adjustment — its balanced top-1 (the competition's tiebreak) and the bits its top-1 carries."""

    context: Score
    word: Score
    dense: Score
    novel: Score
    top1: dict[float, float]
    bits: dict[float, float]
    coverage: Coverage


@dataclass(frozen=True)
class Epoch:
    """One validation pass over every domain, the macro over domains' unseen spans that selects, its interval, and
    the macro top-1 and bits at the adjustment the macro selected — the one a submission is written at."""

    epoch: int
    loss: float
    domains: dict[str, Report]
    macro: Score
    interval: Interval
    top1: float
    bits: float
    fit: float | None = None
    """The context head's balanced top-k on a fixed subset of training words at `tau` (`Telemetry`), None unmeasured."""
    shuffled: float | None = None
    """The macro at `tau` with every word's MEG window swapped for another's of its domain (`Telemetry`)."""

    @property
    def tau(self) -> float:
        return self.macro.tau

    def steering(self) -> dict[str, float | None]:
        """The fit-versus-validation gap and the neural share: the macro less what shuffled MEG still earns."""
        neural = None if self.shuffled is None else self.macro.bacc10 - self.shuffled
        gap = None if self.fit is None else self.fit - self.macro.bacc10
        return {"val": self.macro.bacc10, "fit": self.fit, "gap": gap, "shuffled": self.shuffled, "neural": neural}

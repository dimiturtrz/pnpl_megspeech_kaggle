"""Runs grouped by source, and the rulers that carve val and the local test out of each source's sessions."""

import itertools
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import override

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int

from megspeech.data.raw.runs import Partition, Run
from megspeech.data.stratify import SEED, Picked, Stratify

READ = frozenset({"TIMIT", "MOCHATIMIT"})
"""Tasks of read sentences, the Kaggle test's domain; the rest is narrative (Sherlock, TheMoth)."""


class Sources:
    """Runs grouped by source, the `stratum` of their task, each group in key order: the unit every per-source pick
    (val's carving, the training share, the crossfit folds) is drawn within."""

    @staticmethod
    def stratum(task: str) -> str:
        """The task without its book number: `Sherlock3` reads as `Sherlock`."""
        return re.sub(r"\d+$", "", task)

    @classmethod
    def indices(cls, names: "pd.Series[str]") -> Int[np.ndarray, " n"]:
        """Each run `name`'s source as an index into the sorted sources present."""
        return pd.factorize(Run.tasks(names).map(cls.stratum), sort=True)[0].astype(np.int64)

    @classmethod
    def reads(cls, names: "pd.Series[str]") -> Bool[np.ndarray, " n"]:
        """Whether each run `name`'s task is read sentences (`READ`) rather than narrative."""
        return Run.tasks(names).map(cls.stratum).isin(READ).to_numpy()

    @classmethod
    def of(cls, run: Run) -> tuple[str, tuple[str, int, int]]:
        """`run`'s source and recording key, the order every group is in."""
        return cls.stratum(run.key[0]), run.key

    @classmethod
    def grouped(cls, runs: Iterable[Run]) -> list[tuple[Run, ...]]:
        """`runs` by source, sources in sorted order."""
        ordered = sorted(runs, key=cls.of)
        return [tuple(group) for _, group in itertools.groupby(ordered, key=lambda r: cls.of(r)[0])]

    @classmethod
    def dealt(cls, runs: Sequence[Run], folds: int) -> dict[str, int]:
        """Each run's name to its fold of `folds`: a source's distinct recording keys, sorted, dealt round-robin, every
        run (subject) of a key in its fold, so neighbouring words of a session never straddle folds."""
        fold: dict[str, int] = {}
        for members in cls.grouped(runs):
            dealt = {key: at % folds for at, key in enumerate(sorted({r.key for r in members}))}
            fold |= {r.name: dealt[r.key] for r in members}
        return fold

    @classmethod
    def thinned(cls, runs: Sequence[Run], share: float) -> list[Run]:
        """`runs` cut to `share` of each source's sessions: the leading `ceil(share * n)` of a seeded permutation of the
        source's sessions, so the share's subsets nest; every run when the share is 1."""
        kept: set[str] = set()
        for sessions in cls.grouped(runs):
            order = np.random.default_rng(SEED).permutation(len(sessions))
            kept |= {sessions[i].name for i in order[: math.ceil(share * len(sessions))]}
        return [r for r in runs if r.name in kept]


class Refit(StrEnum):
    """Which held-out sessions a final fit of an already selected recipe also trains on; whatever trains no longer
    scores. `VAL` trains val, `ALL` the local test too: subject 0's, and an adapted model's Broad ruler session
    (`DataConfig.adapted`)."""

    NONE = "none"
    VAL = "val"
    ALL = "all"

    @override
    @classmethod
    def _missing_(cls, value: object) -> "Refit | None":
        """The retired boolean `refit`: true trained val, false nothing."""
        return {True: cls.VAL, False: cls.NONE}.get(value) if isinstance(value, bool) else None


class Val(StrEnum):
    """Which sessions are VAL, selecting epoch and tau; the labelled sessions that report are the LOCAL TEST. `CARVED`:
    10% of each source's subject-0 training sessions, stratified (`Stratify`: source, sentence length, word
    diversity); the local test is pnpl's validation and test partitions pooled. `OFFICIAL`, the legacy full-data
    models': val is pnpl's validation partition, the local test its test partition. `BALANCED`, the retired
    80/10/10 ruler's runs only (a retired `ruler` maps them here): the carved test half stays out of training and
    reports, val and the local test each add pnpl's own partition (megspeech-7gf.122)."""

    CARVED = "carved"
    OFFICIAL = "official"
    BALANCED = "balanced"

    @property
    def carving(self) -> "Carving":
        """What this ruler holds out, and where each held-out session scores."""
        return CARVINGS[self]


@dataclass(frozen=True)
class Carving:
    """A `Val`'s ruler: of each source's sessions, the `share` carved out as val; pnpl's partitions val holds (`held`,
    which train under a val refit) and the local test holds (`tested`); whether `split`'s test half is withheld from
    training and scored as local test (`tests_carved`)."""

    share: float
    held: tuple[Partition, ...]
    tested: tuple[Partition, ...]
    tests_carved: bool = False

    @property
    def stratified(self) -> bool:
        """Whether domains are the sources, as the carved sessions are drawn from them."""
        return self.share > 0

    def picks(self, deep: Sequence[Run]) -> list[tuple[tuple[Run, ...], Picked]]:
        """Each source of `deep` with the sessions `Stratify` holds out of its `n`, `round(share * n)` each."""
        return [(source, Stratify.split(source, round(self.share * len(source)))) for source in Sources.grouped(deep)]


CARVINGS = {
    Val.CARVED: Carving(0.1, held=(), tested=(Partition.TEST, Partition.VALIDATION)),
    Val.OFFICIAL: Carving(0.0, held=(Partition.VALIDATION,), tested=(Partition.TEST,)),
    Val.BALANCED: Carving(0.1, held=(Partition.VALIDATION,), tested=(Partition.TEST,), tests_carved=True),
}

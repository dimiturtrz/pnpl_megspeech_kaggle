"""What a training run reads: which runs train and at what weight, from which cache, cut how."""

import itertools
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Int

from core.paths import DataRoot
from core.tree import Retired, Tree
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED, EVERY, UNALIGNED, Alignment, Geometry, Projection
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import DECIMATE, SFREQ
from megspeech.data.raw.runs import DEEP, Partition, Run
from megspeech.data.spans.grouping import Grouping
from megspeech.data.stratify import SEED, Picked, Stratify

METRICS = "metrics.json"
"""The record a finished run leaves in its directory, its config under `config`."""

READ = frozenset({"TIMIT", "MOCHATIMIT"})
"""Tasks of read sentences, the Kaggle test's domain; the rest is narrative (Sherlock, TheMoth)."""

ADAPTED = "Sherlock1"
"""The task whose sessions 11 and 12 subjects 1-32 each hear once."""

NARRATOR = "Sherlock"
"""The stratum read by one voice throughout; every TheMoth story and read sentence has its own speaker."""


class Refit(StrEnum):
    """Which held-out sessions a final fit of an already selected recipe also trains on; whatever trains no longer
    scores. `VAL` trains val, `ALL` the local test too, subject 0's only."""

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
    def share(self) -> float:
        """Of each source's sessions, the share carved out as val."""
        return {self.CARVED: 0.1, self.OFFICIAL: 0.0, self.BALANCED: 0.1}[self]

    @property
    def stratified(self) -> bool:
        """Whether domains are the sources, as the carved sessions are drawn from them."""
        return self is not Val.OFFICIAL

    @property
    def official_val(self) -> bool:
        """Whether pnpl's validation partition is val (and so trains under a val refit) rather than the local test."""
        return self is not Val.CARVED

    @property
    def tests_carved(self) -> bool:
        """Whether `split`'s test half is withheld from training and scored as local test."""
        return self is Val.BALANCED


BALANCED_RULER = {"val": Val.BALANCED}
STRIDED_RULER = "this run trained under a retired ruler (strided share or extra validation runs)"


@dataclass(frozen=True)
class DataConfig:
    RETIRED: ClassVar[dict[str, Retired]] = {
        "extra": Retired(0, STRIDED_RULER),
        "share": Retired(0.0, STRIDED_RULER, {0.1: BALANCED_RULER}, {"stratify": True}),
        "stratify": Retired(default=False, reason=STRIDED_RULER, becomes={True: {}}),
        "ruler": Retired("official", becomes={"balanced": BALANCED_RULER}),
        "standardised": Retired(
            default=False, reason="per-recording robust scaling (standardised) is retired, killed by megspeech-7gf.25.4"
        ),
        "lead": Retired(0.0, "reading ahead of each onset (lead) is retired, killed by megspeech-7gf.38"),
        "baseline": Retired(0.0, "a pre-response centring stretch (baseline) is retired by megspeech-7gf.157.5"),
    }
    LEGACY: ClassVar[dict[str, object]] = {"val": Val.OFFICIAL}

    tasks: tuple[str, ...] = ()
    """Training restricted to these tasks' runs; empty trains on every task."""
    read_weight: float = 1.0
    """Sampling weight of read-sentence runs (TIMIT, MOCHA-TIMIT) — the Kaggle test's domain — against narrative
    ones."""
    narrator_weight: float = 1.0
    """Sampling weight of runs in `NARRATOR`'s one voice (112 of sub-0's 156 training runs) against the
    multi-speaker rest: TheMoth validation is a voice training never heard, as are the Kaggle test's read sentences."""
    repeat_power: float = 0.0
    """A span heard `k` times in training is drawn with weight `k ** -repeat_power`; 1 draws each sentence once."""
    grouping: Grouping = Grouping.SENTENCE
    """`sentence`: every word of each PNPL sentence; `gap`: vocabulary words only, spans by onset gap."""
    refit: Refit = Refit.NONE
    """Train on val (and with `ALL` the local test) too, for a final fit of an already selected recipe; what trained
    scores words training heard, so it can no longer select or report."""
    window: float = 1.0
    """Seconds of MEG from each word's onset; a model is scored and submitted at the window it trained on."""
    train_share: float = 1.0
    """Of each source's training sessions the ruler leaves, the share that train: whole sessions, the same seeded order
    per source so a smaller share is a subset of a larger one (the data-scaling curve, megspeech-7gf.104)."""
    val: Val = Val.CARVED
    """Which sessions are withheld as val (selects epoch and tau) and which are the local test (reports); see `Val`."""
    delay: float = 0.0
    """Seconds after each word's onset its `window` starts: 0.2 reads the recognition stage past the acoustic one."""
    segment: float = 0.0
    """Seconds of each span's continuous MEG from its first onset that a batch holds in place of word windows, stopped
    at the span's sentence epoch end like every window: a long-context encoder (`context.encoder spanned`) reads all
    its words' features out of that one signal. 0 batches word windows. The Kaggle test holds only each sentence epoch
    (at most 8.7 s), so more adds zeros, never signal."""
    decimate: int = DECIMATE
    """The 250 Hz recordings' decimation factor: which continuous cache training reads (`RunCache.under`)."""
    folds: int = 1
    """Cross-fitting: each source's subject-0 training sessions that test leaves, in key order, dealt round-robin into
    `folds` folds (`folded`); 1 keeps the ruler's validation pick. Every session trains `folds - 1` of the members, and
    each member's validation is the fold it never saw (megspeech-7gf.85)."""
    span_lag: float = -1.0
    """Seconds past the next word's onset where each word's MEG window is cut and zero-padded, the padding being the
    mask: 0 reads the word's true span only, 0.2 lets the response to the next word's onset start; negative reads the
    fixed `window` (megspeech-7gf.98)."""
    fold: int = 0
    """The fold held out as validation when `folds` > 1; the members over every fold ensemble into one."""
    crossfit: int = 1
    """Cross-fitted distillation (megspeech-7gf.130): the training recordings' keys, in order within each source, dealt
    round-robin into this many folds (`fold_of`). Unlike `folds`, val and the local test are untouched."""
    dropped: int = -1
    """The `crossfit` fold removed from training, so a teacher scores it out of sample; -1 trains on every fold."""
    subjects: tuple[int, ...] = (DEEP,)
    """Subjects whose training runs train; validation is always subject 0's, and pooled runs train only as `adapt`."""
    adapt: int = 0
    """The Sherlock1 session (11 or 12) of `subjects` 1-32 that also trains, as labelled Broad data allows; 0 trains
    none. Its text is subject 0's validation (11) or local-test (12) text, so such a model is never a Deep model."""
    adapt_context: bool = False
    """Whether the `adapt` words also train the context head. Off, they feed their spans as context input and train
    the word head and every per-window term, but the context head is not scored on them: all 32 subjects heard the
    same text, so N copies of it make the span transformer memorise it."""
    context_subjects: tuple[int, ...] = ()
    """Subjects whose words the context head (and its distillation) is scored on; empty scores every subject. The
    others still train the word head and every per-window term and still feed their spans as context input, but
    the context head cannot memorise a story they repeat (megspeech-7gf.152)."""
    features: str = ""
    """Features a frozen encoder saved under models/features/<name> (`megspeech.data.batching.split`), read in
    place of every split's MEG; `a+b` joins several encoders' per word (megspeech-7gf.77); empty reads the MEG."""
    cleaning: Cleaning = Cleaning.NONE
    """Artifacts removed from every recording before it is cached (`megspeech.data.cache.cleaning`): which cache
    training and the Kaggle test read."""
    aligned: bool = False
    """Every recording, subject 0's own included, re-coloured to subject 0's mean sensor covariance (`Alignment`) in
    training, validation and Kaggle test alike: one sensor geometry the model learns in and every subject meets it
    in."""
    subspace: bool = False
    """Every recording kept as its coordinates in subject 0's Maxwell signal subspace (`Projection`, ~75 components
    of 306 sensors): the same signal in a quarter of the device cache, the model reading as many channels."""
    components: int = 0
    """The leading components of that subspace a `subspace` recording keeps, largest variance first; 0 keeps every
    one. Fewer inputs than the signal's rank cut the input capacity a small corpus must fit (99% of variance in ~57)."""
    regions: tuple[str, ...] = ()
    """MNE's Vectorview selections (`Left-temporal`, ...) whose sensors alone every recording keeps, in subject 0's
    units; empty keeps all 306. The early (50-150 ms) word-locked field peaks over the temporal lobes: 14 of its top
    20 sensors against 5.1 by chance (n=765256 onsets, megspeech-7gf.77), the acoustic expert's spatial prior."""

    def __post_init__(self) -> None:
        if not 0 < self.train_share <= 1:
            raise ValueError(f"train_share {self.train_share} is not in (0, 1]")
        if not 0 <= self.fold < self.folds:
            raise ValueError(f"fold {self.fold} is not one of the {self.folds} folds")
        if not -1 <= self.dropped < self.crossfit or (self.dropped >= 0 and self.crossfit == 1):
            raise ValueError(f"dropped fold {self.dropped} is not one of the {self.crossfit} crossfit folds")
        if self.subspace and self.regions:
            raise ValueError("a recording keeps either subspace components or region sensors, not both")
        if self.segment < 0 or (self.segment and self.features):
            raise ValueError(f"segment {self.segment} must be 0 or positive, and is read from MEG, not features")

    @property
    def projected(self) -> bool:
        """Whether recordings load as something other than the 306 sensors: subspace components or region sensors."""
        return self.subspace or bool(self.regions)

    @classmethod
    def saved(cls, directory: Path) -> "DataConfig":
        """The data config a run in `directory` recorded; a record without one, or a flat one, falls to defaults."""
        config = json.loads((directory / METRICS).read_text())["config"]
        return Tree(cls).load(config.get("data", {}), saved=True)

    @classmethod
    def partitioned(cls, directories: Sequence[Path]) -> list[list[Path]]:
        """`directories` split by the `READING` each run trained at, each group in first-seen order: the members that
        can share one set of test windows and one prior, which `shared` requires of all of them."""
        groups: dict[tuple[object, ...], list[Path]] = {}
        for directory in directories:
            saved = cls.saved(directory)
            groups.setdefault(tuple(getattr(saved, name) for name in READING), []).append(directory)
        return list(groups.values())

    @classmethod
    def shared(cls, directories: Sequence[Path], grouping: Grouping) -> "DataConfig":
        """The `READING` every run in `directories` trained at, spans cut by `grouping`: members that read MEG
        differently cannot share one set of Kaggle test windows, nor members trained on different runs one prior. Bagged
        members leaving out different folds together trained on every run, so they share an unbagged config."""
        saved = [cls.saved(directory) for directory in directories]
        readings = {tuple(getattr(config, name) for name in READING) for config in saved}
        if len(readings) != 1:
            raise ValueError(f"runs trained at {READING} {sorted(readings)}; an ensemble needs one")
        folds = {config.fold for config in saved}
        bagged = saved[0] if len(folds) == 1 else replace(saved[0], folds=1, fold=0)
        return replace(bagged, grouping=grouping)

    @property
    def rate(self) -> float:
        """Samples per second of the cache training reads."""
        return SFREQ / self.decimate

    def to_samples(self, seconds: float) -> int:
        """`seconds` in samples at the cache's `rate`."""
        return round(seconds * self.rate)

    @property
    def samples(self) -> int:
        """The window in samples at the cache's `rate`."""
        return self.to_samples(self.window)

    def cut(self) -> tuple[int, int]:
        """The frames each word's window spans and its start's offset from the word's onset, at the cache's `rate`."""
        return self.samples, self.to_samples(self.delay)

    def reading(self, windows: Windows) -> Windows:
        """`windows` cut as this config reads them: `window` from `delay` after each onset, centred on itself."""
        length, offset = self.cut()
        return replace(self.spanned(windows), length=length, offset=offset, segment=self.to_samples(self.segment))

    def spanned(self, windows: Windows) -> Windows:
        """`windows` stopped `span_lag` after each word's next onset, never before its own first sample; as they are
        when `span_lag` is negative or no next onset is known."""
        if self.span_lag < 0 or windows.following is None:
            return windows
        cut = windows.following + self.to_samples(self.span_lag)
        return replace(windows, end=torch.minimum(windows.end, cut).clamp(min=windows.at + 1))

    @property
    def recoloured(self) -> bool:
        """Whether recordings are re-coloured (`alignment`): every one when `aligned`, the other subjects' whenever
        they train, so evaluation reads them in the geometry training did."""
        return self.aligned or any(subject != DEEP for subject in self.subjects)

    def run_cache(self, root: DataRoot) -> RunCache:
        """The run cache training reads, at `decimate`."""
        return RunCache.under(root.processed(), self.decimate, self.cleaning)

    def geometry(self, root: DataRoot, device: torch.device | str, *, aligned: bool = False) -> Geometry:
        """The `Geometry` recordings load in: re-coloured to subject 0's `alignment` when `recoloured` (or `aligned`
        asks for it), kept in its `Projection` when `projected`; the reference left unbuilt when neither reads it."""
        recolour = aligned or self.recoloured
        if not (recolour or self.projected):
            return AS_RECORDED
        alignment = self.alignment(root, device)
        return Geometry(alignment if recolour else None, self.projection(alignment, root))

    def projection(self, alignment: Alignment, root: DataRoot) -> Projection | None:
        """The `Projection` recordings are kept in: `regions`' sensors, `subspace`'s components, or none."""
        if self.regions:
            return alignment.selection(self.regions, root)
        return alignment.projection(self.components) if self.subspace else None

    def alignment(self, root: DataRoot, device: torch.device | str) -> Alignment:
        """Subject 0's sensor reference in this config's cache, from every `EVERY`th of its training runs: every
        run re-coloured to it when `aligned`, else every other subject's."""
        cache = self.run_cache(root)
        reference = [run for run in self.trained(Run.discover(root.raw())) if run.subject == DEEP][::EVERY]
        signals = [np.load(cache.path(run.name), mmap_mode="r") for run in reference]
        return Alignment.of(signals, device, () if self.aligned else UNALIGNED)

    def weight(self, run: Run) -> float:
        """`run`'s sampling weight: `read_weight` for a read-sentence task, `narrator_weight` for `NARRATOR`'s
        voice, 1 for the other narrative runs."""
        weights = {**dict.fromkeys(READ, self.read_weight), NARRATOR: self.narrator_weight}
        return weights.get(self.stratum(run.key[0]), 1.0)

    @property
    def refitted(self) -> tuple[Partition, ...]:
        """The official partitions of subject 0 that train too: val's under `OFFICIAL`, both under `Refit.ALL`."""
        official = (Partition.VALIDATION,) if self.val.official_val else ()
        return {Refit.NONE: (), Refit.VAL: official, Refit.ALL: (Partition.VALIDATION, Partition.TEST)}[self.refit]

    def trains_on(self, run: Run) -> bool:
        """Whether `run` trains: a `TRAIN` run of a subject in `subjects`, or one of the `refitted` partitions of
        subject 0; the `adapt` session of every other subject in `subjects`; never another subject's other runs."""
        adapted = run.subject in self.subjects and run.subject != DEEP and run.key == (ADAPTED, self.adapt, 1)
        reads = (
            (run.partition is Partition.TRAIN and run.subject in self.subjects)
            or (run.partition in self.refitted and run.subject == DEEP and DEEP in self.subjects)
            or adapted
        )
        return reads and (not self.tasks or run.key[0] in self.tasks)

    def muted(self, names: "pd.Series[str]") -> Bool[np.ndarray, " n"]:
        """Whether each run `name`'s words keep the context head's label from training: the `adapt` runs of subjects
        1-32, unless `adapt_context`, and every run of a subject outside `context_subjects` when it is set."""
        subjects = Run.subjects(names)
        adapted = Run.matching(names, (ADAPTED, self.adapt, 1)) & (subjects != DEEP)
        outside = ~np.isin(subjects, self.context_subjects) if self.context_subjects else np.zeros(len(names), bool)
        return (adapted & (not self.adapt_context)) | outside

    @staticmethod
    def stratum(task: str) -> str:
        """The task without its book number: `Sherlock3` reads as `Sherlock`."""
        return re.sub(r"\d+$", "", task)

    @classmethod
    def sources(cls, names: "pd.Series[str]") -> Int[np.ndarray, " n"]:
        """Each run `name`'s dataset (`stratum` of its task) as an index into the sorted datasets present."""
        strata = Run.tasks(names).map(cls.stratum)
        return pd.factorize(strata, sort=True)[0].astype(np.int64)

    @classmethod
    def reads(cls, names: "pd.Series[str]") -> Bool[np.ndarray, " n"]:
        """Whether each run `name`'s task is read sentences (`READ`) rather than narrative."""
        return Run.tasks(names).map(cls.stratum).isin(READ).to_numpy()

    def source(self, run: Run) -> tuple[str, tuple[str, int, int]]:
        return self.stratum(run.key[0]), run.key

    def split(self, runs: Sequence[Run]) -> dict[Partition, list[Run]]:
        """Subject 0's training sessions `share` holds out for validation and test, `round(share * n)` of each source's
        `n` each, picked by `Stratify`."""
        deep = sorted((r for r in runs if r.partition is Partition.TRAIN and r.subject == DEEP), key=self.source)
        sources = [tuple(group) for _, group in itertools.groupby(deep, key=lambda r: self.source(r)[0])]
        held_out = [
            self.folded(source, Stratify.split(source, round(self.val.share * len(source)))) for source in sources
        ]
        return {
            partition: [run for chosen in held_out for run in chosen[at]]
            for at, partition in enumerate((Partition.VALIDATION, Partition.TEST))
        }

    def folded(self, source: tuple[Run, ...], picked: Picked) -> Picked:
        """`picked`; when `folds` > 1, its validation the `fold`th of every `folds` sessions its test leaves."""
        if self.folds == 1:
            return picked
        _, test = picked
        tested = {r.name for r in test}
        return [r for r in source if r.name not in tested][self.fold :: self.folds], test

    def carved(self, runs: Sequence[Run]) -> list[Run]:
        """Subject 0's sessions `Val.CARVED` holds out as val: `split`'s validation half, the fold when `folds` > 1;
        none under `OFFICIAL`. The one place val's membership is chosen."""
        return self.split(runs)[Partition.VALIDATION]

    def tested_carved(self, runs: Sequence[Run]) -> list[Run]:
        """Subject 0's sessions `Val.BALANCED` also holds out as local test: `split`'s test half; none otherwise."""
        return self.split(runs)[Partition.TEST] if self.val.tests_carved else []

    def trained(self, runs: Sequence[Run]) -> list[Run]:
        """The runs training reads: `trains_on`, less the carved val sessions unless refitting."""
        pool = self.pooled(runs)
        if self.dropped < 0:
            return pool
        fold = self.fold_of(runs, self.crossfit)
        return [r for r in pool if fold[r.name] != self.dropped]

    def pooled(self, runs: Sequence[Run]) -> list[Run]:
        """`trained` before any `dropped` fold leaves."""
        excluded = {r.name for r in self.carved(runs)} if self.refit is Refit.NONE else set()
        excluded |= {r.name for r in self.tested_carved(runs)}
        return self.thinned([r for r in runs if self.trains_on(r) and r.name not in excluded])

    def fold_of(self, runs: Sequence[Run], folds: int) -> dict[str, int]:
        """Each pooled run's name to its fold of `folds`: a source's distinct recording keys, sorted, dealt round-robin,
        every run (subject) of a key in its fold, so neighbouring words of a session never straddle folds."""
        pool = sorted(self.pooled(runs), key=self.source)
        fold: dict[str, int] = {}
        for _, group in itertools.groupby(pool, key=lambda r: self.source(r)[0]):
            members = list(group)
            dealt = {key: at % folds for at, key in enumerate(sorted({r.key for r in members}))}
            fold |= {r.name: dealt[r.key] for r in members}
        return fold

    def thinned(self, runs: Sequence[Run]) -> list[Run]:
        """`runs` cut to `train_share` of each source's sessions: the leading `ceil(share * n)` of a seeded permutation
        of the source's sessions, so the share's subsets nest; every run when the share is 1."""
        if self.train_share == 1:
            return list(runs)
        ordered = sorted(runs, key=self.source)
        kept: set[str] = set()
        for _, group in itertools.groupby(ordered, key=lambda r: self.source(r)[0]):
            sessions = list(group)
            order = np.random.default_rng(SEED).permutation(len(sessions))
            kept |= {sessions[i].name for i in order[: math.ceil(self.train_share * len(sessions))]}
        return [r for r in runs if r.name in kept]

    def domain(self, run: Run) -> str:
        """The domain `run` is scored in: its source under `CARVED` or `folds`, else its task."""
        return self.stratum(run.key[0]) if self.val.stratified or self.folds > 1 else run.key[0]

    def official(self, runs: Sequence[Run], *partitions: Partition) -> list[Run]:
        """Subject 0's runs of the official `partitions`; other subjects' never count."""
        return [r for r in runs if r.partition in partitions and r.subject == DEEP]

    def scored(self, runs: Sequence[Run], partition: Partition) -> dict[str, list[Run]]:
        """The domains of val (`Partition.VALIDATION`) or of the local test (`Partition.TEST`), under `domain`: val is
        the carved sessions, or pnpl's validation partition under `OFFICIAL`; the local test pnpl's test partition,
        with its validation partition too under `CARVED`, and under `BALANCED` the
        carved test half as well."""
        if partition is Partition.VALIDATION:
            chosen = [*self.official(runs, *([partition] if self.val.official_val else [])), *self.carved(runs)]
        else:
            pooled = [] if self.val.official_val else [Partition.VALIDATION]
            chosen = [*self.official(runs, partition, *pooled), *self.tested_carved(runs)]
        names = sorted({self.domain(r) for r in chosen})
        return {name: [r for r in chosen if self.domain(r) == name] for name in names}

    def tested(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        """The local test's domains, which report."""
        return self.scored(runs, Partition.TEST)

    def held(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        """Val's domains' runs, which select epoch and adjustment."""
        return self.scored(runs, Partition.VALIDATION)

    def carried(self, table: pd.DataFrame) -> pd.DataFrame:
        """The words of `table` the grouping carries, at this config's rate (`Grouping.carried`)."""
        return RunCache.carried(table, self.grouping, self.rate)

    def ruled(self, runs: Sequence[Run], partition: Partition) -> dict[str, list[Run]]:
        """The domains `partition` is scored over: validation's (`held`) or test's (`tested`)."""
        return {Partition.VALIDATION: self.held, Partition.TEST: self.tested}[partition](runs)


UNREAD = frozenset({"grouping", "fold", "crossfit", "dropped"})
READING = tuple(field.name for field in fields(DataConfig) if field.name not in UNREAD)
"""The `DataConfig` fields an ensemble's members must agree on: how each reads the MEG, which runs trained it and at
what weight. The rest are `grouping`, the scorer's own, and `fold`, `crossfit` and `dropped`, which members differ in
by design."""

"""What a training run reads: which runs train and at what weight, from which cache, cut how."""

import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED, EVERY, UNALIGNED, Alignment, Geometry, Projection
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import DECIMATE, SFREQ
from megspeech.data.raw.runs import DEEP, Partition, Run
from megspeech.data.spans.spans import Grouping
from megspeech.data.stratify import SEED, Picked, Stratify

METRICS = "metrics.json"
"""The record a finished run leaves in its directory, its config under `config`."""

READ = frozenset({"TIMIT", "MOCHATIMIT"})
"""Tasks of read sentences, the Kaggle test's domain; the rest is narrative (Sherlock, TheMoth)."""

NARRATOR = "Sherlock"
"""The stratum read by one voice throughout; every TheMoth story and read sentence has its own speaker."""


class LocalRuler(StrEnum):
    """The two local rulers; their scores never compare. `OFFICIAL`: pnpl's labelled partitions, official validation
    selects epoch and adjustment, official test reports. `BALANCED`: a stratified 80/10/10 session split of subject 0's
    training sessions (`Stratify`), each source's validation selecting and test reporting, the official runs joining
    theirs: ~30k scored words per partition against the official ~2.5k, whose interval spans +-0.03 (megspeech-7gf.81,
    megspeech-7gf.103)."""

    OFFICIAL = "official"
    BALANCED = "balanced"

    @property
    def share(self) -> float:
        """Of each source's sessions, the share held out for validation and as much for test."""
        return {self.OFFICIAL: 0.0, self.BALANCED: 0.1}[self]

    @classmethod
    def migrated(cls, record: Mapping[str, object]) -> dict[str, object]:
        """A saved data config under the retired `extra`/`share`/`stratify` fields as one under `ruler`; the retired
        strided split and extra validation domains are refused."""
        kept = {k: v for k, v in record.items() if k not in {"extra", "share", "stratify"}}
        share, stratified = float(str(record.get("share", 0.0))), bool(record.get("stratify", False))
        if int(str(record.get("extra", 0))) or (share and not stratified):
            raise ValueError("this run trained under a retired ruler (strided share or extra validation runs)")
        ruled: dict[str, object] = {"ruler": cls.BALANCED} if share else {}
        return kept | ruled


@dataclass(frozen=True)
class DataConfig:
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
    refit: bool = False
    """Train on the validation runs too, for a final fit of an already selected recipe; validation then scores
    words training heard, so it can no longer select."""
    window: float = 1.0
    """Seconds of MEG from each word's onset; a model is scored and submitted at the window it trained on."""
    standardised: bool = False
    """Each recording — training or validation run, Kaggle test archive — robust-scaled per channel by its own median
    and IQR before the model's scaler. Across the 162 runs a channel's IQR over its median run spans 0.78-1.28 (p5-p95);
    the Kaggle test sentence archive's sits at 0.90."""
    train_share: float = 1.0
    """Of each source's training sessions the ruler leaves, the share that train: whole sessions, the same seeded order
    per source so a smaller share is a subset of a larger one (the data-scaling curve, megspeech-7gf.104)."""
    ruler: LocalRuler = LocalRuler.OFFICIAL
    """The local ruler the model trains under: which sessions are withheld for validation (selects epoch and tau) and
    test (reports); see `LocalRuler`."""
    baseline: float = 0.0
    """Seconds from each window's start its centring mean is taken over; 0 centres on the whole window. The
    reference pipeline baselines on 0-0.5 s, the pre-response stretch, so the word's evoked field is kept whole."""
    lead: float = 0.0
    """Seconds of MEG before each word's onset, read ahead of its `window`: never before its sentence's first onset,
    as a Kaggle test sentence epoch starts there."""
    delay: float = 0.0
    """Seconds after each word's onset its `window` starts: 0.2 reads the recognition stage past the acoustic one."""
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
    subjects: tuple[int, ...] = (DEEP,)
    """Subjects whose training runs train; validation is always subject 0's, and pooled runs never train."""
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
        if self.subspace and self.regions:
            raise ValueError("a recording keeps either subspace components or region sensors, not both")

    @property
    def projected(self) -> bool:
        """Whether recordings load as something other than the 306 sensors: subspace components or region sensors."""
        return self.subspace or bool(self.regions)

    @classmethod
    def saved(cls, directory: Path) -> "DataConfig":
        """The data config a run in `directory` recorded; a record without one, or a flat one, falls to defaults."""
        config = json.loads((directory / METRICS).read_text())["config"]
        return Tree(cls).load(LocalRuler.migrated(config.get("data", {})))

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

    def reading(self, windows: Windows) -> Windows:
        """`windows` cut as this config reads them: `lead` before each onset, `window` from `delay` after it, centred
        on `baseline` (0 centres on the whole window)."""
        lead = self.to_samples(self.lead)
        offset = self.to_samples(self.delay) - lead
        spanned = self.spanned(windows)
        return replace(spanned, length=self.samples + lead, offset=offset, baseline=self.to_samples(self.baseline))

    def spanned(self, windows: Windows) -> Windows:
        """`windows` stopped `span_lag` after each word's next onset, never before its own first sample; as they are
        when `span_lag` is negative or no next onset is known."""
        if self.span_lag < 0 or windows.following is None:
            return windows
        cut = windows.following + self.to_samples(self.span_lag)
        return replace(windows, end=torch.minimum(windows.end, cut).clamp(min=windows.at + 1))

    def under(self, ruler: "LocalRuler") -> "DataConfig":
        """This config scored under `ruler`: the official one needs no carved sessions, so it also reads a model trained
        on the balanced split; the balanced one only reads a model trained on it (megspeech-7gf.103)."""
        if ruler is self.ruler:
            return self
        if ruler is not LocalRuler.OFFICIAL:
            raise ValueError(f"a model trained under the {self.ruler} ruler cannot be scored under {ruler}")
        return replace(self, ruler=ruler, folds=1, fold=0)

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

    def trains_on(self, run: Run) -> bool:
        trained = (Partition.TRAIN, Partition.VALIDATION) if self.refit else (Partition.TRAIN,)
        return (
            run.partition in trained and run.subject in self.subjects and (not self.tasks or run.key[0] in self.tasks)
        )

    @staticmethod
    def stratum(task: str) -> str:
        """The task without its book number: `Sherlock3` reads as `Sherlock`."""
        return re.sub(r"\d+$", "", task)

    def source(self, run: Run) -> tuple[str, tuple[str, int, int]]:
        return self.stratum(run.key[0]), run.key

    def split(self, runs: Sequence[Run]) -> dict[Partition, list[Run]]:
        """Subject 0's training sessions `share` holds out for validation and test, `round(share * n)` of each source's
        `n` each, picked by `Stratify`."""
        deep = sorted((r for r in runs if r.partition is Partition.TRAIN and r.subject == DEEP), key=self.source)
        sources = [tuple(group) for _, group in itertools.groupby(deep, key=lambda r: self.source(r)[0])]
        held_out = [
            self.folded(source, Stratify.split(source, round(self.ruler.share * len(source)))) for source in sources
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

    def trained(self, runs: Sequence[Run]) -> list[Run]:
        """The runs training reads: `trains_on`, less the held-out test runs, and less the held-out validation runs
        unless refitting on validation."""
        held_out = self.split(runs)
        held: list[Run] = [] if self.refit else held_out[Partition.VALIDATION]
        excluded = {r.name for r in [*held, *held_out[Partition.TEST]]}
        return self.thinned([r for r in runs if self.trains_on(r) and r.name not in excluded])

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
        """The domain `run` is scored in: its source once `share` or `folds` holds out sessions, else its task."""
        return self.stratum(run.key[0]) if self.ruler is LocalRuler.BALANCED or self.folds > 1 else run.key[0]

    def scored(self, runs: Sequence[Run], partition: Partition) -> dict[str, list[Run]]:
        """`partition`'s domains: subject 0's official runs of it with the sessions held out for it, under `domain`."""
        chosen = [r for r in runs if r.partition is partition and r.subject == DEEP] + self.split(runs)[partition]
        names = sorted({self.domain(r) for r in chosen})
        return {name: [r for r in chosen if self.domain(r) == name] for name in names}

    def tested(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        return self.scored(runs, Partition.TEST)

    def held(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        """The validation domains' runs, which select epoch and adjustment."""
        return self.scored(runs, Partition.VALIDATION)

    def ruled(self, runs: Sequence[Run], partition: Partition) -> dict[str, list[Run]]:
        """The domains `partition` is scored over: validation's (`held`) or test's (`tested`)."""
        return {Partition.VALIDATION: self.held, Partition.TEST: self.tested}[partition](runs)


READING = tuple(field.name for field in fields(DataConfig) if field.name not in {"grouping", "fold"})
"""The `DataConfig` fields an ensemble's members must agree on: how each reads the MEG, which runs trained it and at
what weight. The rest are `grouping`, the scorer's own, and `fold`, which bagged members differ in by design."""

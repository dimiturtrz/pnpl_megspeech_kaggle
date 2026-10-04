"""What a training run reads: which runs train and at what weight, from which cache, cut how."""

import itertools
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED, EVERY, UNALIGNED, Alignment, Geometry
from megspeech.data.cache.cleaning import Cleaning
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import DECIMATE, SFREQ
from megspeech.data.raw.runs import DEEP, Partition, Run
from megspeech.data.spans.spans import Grouping
from megspeech.data.stratify import Picked, Stratify

METRICS = "metrics.json"
"""The record a finished run leaves in its directory, its config under `config`."""

PHASES = (0.25, 0.75)
"""Where in each stride of a source's sessions the `strided` validation and test sessions sit."""

READ = frozenset({"TIMIT", "MOCHATIMIT"})
"""Tasks of read sentences, the Kaggle test's domain; the rest is narrative (Sherlock, TheMoth)."""

NARRATIVE = ("Sherlock", "TheMoth")
"""The strata the ruler scores: Sherlock's nine books are one, TheMoth's stories another."""

NARRATOR = "Sherlock"
"""The stratum read by one voice throughout; every TheMoth story and read sentence has its own speaker."""


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
    extra: int = 0
    """Whole training sessions per narrative stratum held out as one more validation domain, `<stratum>+`, beside
    the official run: each book's or story's last session first. One official run per stratum leaves the macro's
    interval near +-0.035; `k` more runs of similar size shrink it by about `sqrt(1 + k)`, for `k` runs of the
    ~150 trained."""
    share: float = 0.0
    """Of each source's subject-0 training sessions (`stratum`), the share held out for validation and as much for
    test (`split`); domains are then scored per source, the official runs joining theirs. 0 keeps the official split,
    one run per task. 0.1 holds out 11+11 of Sherlock's 112 sessions, 3+3 of TheMoth's 28 and 1+1 of TIMIT's 14: ~30k
    scored words per partition against the official ~2.5k, whose interval spans +-0.03 (megspeech-7gf.81)."""
    stratify: bool = False
    """Pick each source's `share` by `Stratify`: the sessions whose words best match training's on volume, word rate,
    sentence length, in-vocabulary share, label mix and book mix. Off picks one per stride in key order (`strided`)."""
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
    `folds` folds (`folded`); 1 keeps `share`'s validation pick. Every session trains `folds - 1` of the members, and
    each member's validation is the fold it never saw (megspeech-7gf.85)."""
    fold: int = 0
    """The fold held out as validation when `folds` > 1; the members over every fold ensemble into one."""
    subjects: tuple[int, ...] = (DEEP,)
    """Subjects whose training runs train; validation is always subject 0's, and pooled runs never train."""
    features: str = ""
    """Features a frozen encoder saved under models/features/<name> (`megspeech.data.batching.split`), read in
    place of every split's MEG; empty reads the MEG."""
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

    def __post_init__(self) -> None:
        if not 0 <= self.fold < self.folds:
            raise ValueError(f"fold {self.fold} is not one of the {self.folds} folds")

    @classmethod
    def saved(cls, directory: Path) -> "DataConfig":
        """The data config a run in `directory` recorded; a record without one, or a flat one, falls to defaults."""
        config = json.loads((directory / METRICS).read_text())["config"]
        return Tree(cls).load(config.get("data", {}))

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
        return replace(windows, length=self.samples + lead, offset=offset, baseline=self.to_samples(self.baseline))

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
        asks for it), kept in its `Projection` when `subspace`; the reference left unbuilt when neither reads it."""
        recolour = aligned or self.recoloured
        if not (recolour or self.subspace):
            return AS_RECORDED
        alignment = self.alignment(root, device)
        projection = alignment.projection(self.components) if self.subspace else None
        return Geometry(alignment if recolour else None, projection)

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

    def extras(self, runs: Sequence[Run]) -> list[Run]:
        """The `extra` training runs per narrative stratum held out: every book's last session, then every book's
        second-to-last, books in order."""
        deep = (r for r in runs if r.partition is Partition.TRAIN and r.subject == DEEP)
        train = sorted(deep, key=lambda r: r.key, reverse=True)
        ranked = sorted(
            ((rank, run.key[0]), run)
            for _, book in itertools.groupby(train, key=lambda r: r.key[0])
            for rank, run in enumerate(book)
        )
        return [
            run
            for stratum in NARRATIVE
            for _, run in [entry for entry in ranked if self.stratum(entry[0][1]) == stratum][: self.extra]
        ]

    def source(self, run: Run) -> tuple[str, tuple[str, int, int]]:
        return self.stratum(run.key[0]), run.key

    @staticmethod
    def strided(source: tuple[Run, ...], count: int) -> Picked:
        """In key order (book, then recording date), one session per stride: spans every book and date."""
        val, test = ([source[int((j + phase) * len(source) / count)] for j in range(count)] for phase in PHASES)
        return val, test

    def split(self, runs: Sequence[Run]) -> dict[Partition, list[Run]]:
        """Subject 0's training sessions `share` holds out for validation and test, `round(share * n)` of each source's
        `n` each, `stratify`'d or `strided` from the source's sessions in key order."""
        deep = sorted((r for r in runs if r.partition is Partition.TRAIN and r.subject == DEEP), key=self.source)
        sources = [tuple(group) for _, group in itertools.groupby(deep, key=lambda r: self.source(r)[0])]
        pick = Stratify.split if self.stratify else self.strided
        held_out = [self.folded(source, pick(source, round(self.share * len(source)))) for source in sources]
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
        """The runs training reads: `trains_on`, less the held-out test runs, and less the extra and held-out validation
        runs unless refitting on validation."""
        held_out = self.split(runs)
        held: list[Run] = [] if self.refit else [*self.extras(runs), *held_out[Partition.VALIDATION]]
        excluded = {r.name for r in [*held, *held_out[Partition.TEST]]}
        return [r for r in runs if self.trains_on(r) and r.name not in excluded]

    def domain(self, run: Run) -> str:
        """The domain `run` is scored in: its source once `share` or `folds` holds out sessions, else its task."""
        return self.stratum(run.key[0]) if self.share or self.folds > 1 else run.key[0]

    def scored(self, runs: Sequence[Run], partition: Partition) -> dict[str, list[Run]]:
        """`partition`'s domains: subject 0's official runs of it with the sessions held out for it, under `domain`."""
        chosen = [r for r in runs if r.partition is partition and r.subject == DEEP] + self.split(runs)[partition]
        names = sorted({self.domain(r) for r in chosen})
        return {name: [r for r in chosen if self.domain(r) == name] for name in names}

    def tested(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        return self.scored(runs, Partition.TEST)

    def held(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        """The validation domains' runs (`scored`), the extras under `<stratum>+`."""
        domains = self.scored(runs, Partition.VALIDATION)
        extras = {f"{s}+": [r for r in self.extras(runs) if self.stratum(r.key[0]) == s] for s in NARRATIVE}
        return domains | {name: chosen for name, chosen in extras.items() if chosen}

    def ruled(self, runs: Sequence[Run], partition: Partition) -> dict[str, list[Run]]:
        """The domains `partition` is scored over: validation's (`held`) or test's (`tested`)."""
        return {Partition.VALIDATION: self.held, Partition.TEST: self.tested}[partition](runs)


READING = tuple(field.name for field in fields(DataConfig) if field.name not in {"grouping", "fold"})
"""The `DataConfig` fields an ensemble's members must agree on: how each reads the MEG, which runs trained it and at
what weight. The rest are `grouping`, the scorer's own, and `fold`, which bagged members differ in by design."""

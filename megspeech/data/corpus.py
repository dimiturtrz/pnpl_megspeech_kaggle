"""What a training run reads: which runs train and at what weight, cut from the continuous cache with validation."""

import itertools
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int

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

NARRATIVE = ("Sherlock", "TheMoth")
"""The strata the ruler scores: Sherlock's nine books are one, TheMoth's stories another."""


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
    extra: int = 0
    """Whole training sessions per narrative stratum held out as one more validation domain, `<stratum>+`, beside
    the official run: each book's or story's last session first. One official run per stratum leaves the macro's
    interval near +-0.035; `k` more runs of similar size shrink it by about `sqrt(1 + k)`, for `k` runs of the
    ~150 trained."""
    baseline: float = 0.0
    """Seconds from each window's start its centring mean is taken over; 0 centres on the whole window. The
    reference pipeline baselines on 0-0.5 s, the pre-response stretch, so the word's evoked field is kept whole."""

    @classmethod
    def saved(cls, directory: Path) -> "DataConfig":
        """The data config a run in `directory` recorded; a record without one, or a flat one, falls to defaults."""
        config = json.loads((directory / METRICS).read_text())["config"]
        return Tree(cls).load(config.get("data", {}))

    @classmethod
    def shared(cls, directories: Sequence[Path], grouping: Grouping) -> "DataConfig":
        """The holdout reading — window and standardisation — every run in `directories` trained on, spans cut by
        `grouping`; members that read MEG differently cannot share one set of holdout windows."""
        readings = {
            (saved.window, saved.standardised, saved.extra, saved.baseline) for saved in map(cls.saved, directories)
        }
        if len(readings) != 1:
            raise ValueError(
                f"runs trained at (window, standardised, extra, baseline) {sorted(readings)}; an ensemble needs one"
            )
        window, standardised, extra, baseline = readings.pop()
        return cls(grouping=grouping, window=window, standardised=standardised, extra=extra, baseline=baseline)

    @property
    def samples(self) -> int:
        """The window in samples at the cache's `RATE`."""
        return round(self.window * RATE)

    @property
    def centring(self) -> int:
        """The centring baseline in samples; 0 centres on the whole window."""
        return round(self.baseline * RATE)

    def trains_on(self, run: Run) -> bool:
        trained = (Partition.TRAIN, Partition.VALIDATION) if self.refit else (Partition.TRAIN,)
        return run.partition in trained and (not self.tasks or run.key[0] in self.tasks)

    @staticmethod
    def stratum(task: str) -> str:
        """The task without its book number: `Sherlock3` reads as `Sherlock`."""
        return re.sub(r"\d+$", "", task)

    def extras(self, runs: Sequence[Run]) -> list[Run]:
        """The `extra` training runs per narrative stratum held out: every book's last session, then every book's
        second-to-last, books in order."""
        train = sorted((r for r in runs if r.partition is Partition.TRAIN), key=lambda r: r.key, reverse=True)
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

    def trained(self, runs: Sequence[Run]) -> list[Run]:
        """The runs training reads: `trains_on`, less the extra validation runs unless refitting on validation."""
        withheld = set() if self.refit else {r.name for r in self.extras(runs)}
        return [r for r in runs if self.trains_on(r) and r.name not in withheld]

    def held(self, runs: Sequence[Run]) -> dict[str, list[Run]]:
        """The validation domains' runs: each official validation run under its task, the extras under `<stratum>+`."""
        official = [r for r in runs if r.partition is Partition.VALIDATION]
        domains = {task: [r for r in official if r.key[0] == task] for task in sorted({r.key[0] for r in official})}
        extras = {f"{s}+": [r for r in self.extras(runs) if self.stratum(r.key[0]) == s] for s in NARRATIVE}
        return domains | {name: chosen for name, chosen in extras.items() if chosen}


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
        trained, held = data.trained(runs), data.held(runs)
        reading = [r.name for r in trained] + [r.name for group in held.values() for r in group]
        source = Continuous.load(cache, device, list(dict.fromkeys(reading)), standardised=data.standardised)
        words = cls.words(source.words, data.grouping)
        weights = {r.name: data.read_weight if r.key[0] in READ else 1.0 for r in trained}
        chosen = words[words["run"].isin(weights)]
        weight = chosen["run"].map(weights).to_numpy(dtype=np.float64)
        train = cls.split(source, chosen, data, weight).recurring(data.repeat_power)
        validation = cls.domains(source, words, held, data)
        heard = cls.heard(cache, runs, data)
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
        return table.assign(target=target, token=Corpus.token(table["word"]))[
            grouping.keep(target) & Continuous.inside(table)
        ]

    @staticmethod
    def token(words: pd.Series) -> Int[np.ndarray, " n"]:
        """Each word lower-cased and hashed: the key every per-word table shares with a split's `token`."""
        return pd.util.hash_array(words.astype(str).str.lower().to_numpy()).view(np.int64)

    @staticmethod
    def split(source: Continuous, words: pd.DataFrame, data: DataConfig, weight: Float[np.ndarray, " n"]) -> Split:
        """`words` (rows of `source.words`) as a split whose `data.window`-long windows are gathered from `source`'s
        device signal, each stopped where `data.grouping` bounds it."""
        windows = Windows(
            source.meg,
            source.samples(words["at"]),
            source.samples(data.grouping.bound(words)),
            data.samples,
            baseline=data.centring,
        )
        labels = (words["target"].to_numpy(), words["token"].to_numpy())
        return Split.of(windows, labels, data.grouping.spans(words), weight)

    @classmethod
    def domains(
        cls, source: Continuous, words: pd.DataFrame, held: dict[str, list[Run]], data: DataConfig
    ) -> dict[str, Split]:
        """Each validation domain's `words` (`DataConfig.held`), one split per domain so each is scored alone."""
        chosen = {name: words[words["run"].isin([r.name for r in runs])] for name, runs in held.items()}
        return {name: cls.split(source, rows, data, np.ones(len(rows))) for name, rows in chosen.items()}

    @classmethod
    def heard(cls, cache: Path, runs: Sequence[Run], data: DataConfig) -> pd.DataFrame:
        """Every training-partition word `data.grouping` carries outside the extra validation runs, read from the
        word table without any MEG: the text a novel span must not repeat."""
        withheld = {r.name for r in data.extras(runs)}
        names = [r.name for r in runs if r.partition is Partition.TRAIN and r.name not in withheld]
        return cls.words(Continuous.table(cache, names), data.grouping)

    @staticmethod
    def sequences(heard: pd.DataFrame, grouping: Grouping) -> frozenset[tuple[int, ...]]:
        """The text of every span of `heard` as its token sequence, spans cut run by run."""
        runs = heard.groupby("run", sort=False)
        return Split.seen((run["token"].to_numpy(), grouping.spans(run)) for _, run in runs)

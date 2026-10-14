"""Whether a part-of-speech posterior narrows the 50 candidates of the 50-way decoder (megspeech-7gf.94, -cn2). Each
scored word's tag likelihood ratio of a candidate word w is sum_t p(t | w) p(t | MEG) / p(t): the word's class
distribution in context on the training runs (`megspeech.data.labels.context_tags`, add-one smoothed) against a
posterior over the word's class. A token the tagger could not align has no class and is left out.
The probe's classes are
weighted by frequency (natural) or equally (balanced), the prior p(t) matching. Two arms read the ratio:

- hierarchical: the vocabulary scores plus `lam` times its log, the factorisation log p(w | MEG) + lam log ratio;
- gate: the vocabulary scores with every candidate whose ratio is under the threshold ranked behind all the rest.

The posterior is the stacked probe's, over timing and the context embeddings of the model named by `--embeddings`
(default the decoder's own), or over the word embeddings alone for `--stack word`. The decoder is frozen and the
fusion runs on the CPU. The setting of each arm (`lam`, the threshold) is fit leave-one-domain-out over the official
validation domains: chosen on two, scored on the third, the macro read as the official scorer reads it. The score at
every grid setting over all domains is a diagnostic fit on validation.

    uv run python -m megspeech.evaluation.studies.hierarchy sent40-ioi-ema9998-s0
"""

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.heard import Heard
from megspeech.data.labels.context_tags import ContextTags
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import DEEP, Column, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head, Score, SelectionConfig
from megspeech.evaluation.studies.syntax import FLOOR, Embedded, Probe, Table, Timing
from megspeech.model.checkpoint import Checkpoint
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.subjects import Fallback, SubjectLayer, Unseen

LOG = logging.getLogger(__name__)

GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
"""Fusion weights on the part-of-speech bonus; 0 is the context head alone."""
THRESHOLDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Likelihood-ratio floors of the gate; 0 gates nothing and 1 gates every candidate the tags speak against."""
DEMOTED = 1000.0
"""What a gated candidate's score drops by: more than any logit spread and prior adjustment, so it ranks last."""
STACKS = {"context": ("timing", "meg-context"), "word": ("meg-word",)}
"""The probes whose out-of-fold log-probabilities the stacker reads, by `--stack`."""
OFFICIAL = 0.6445
"""The context head's official macro BAcc@10 for sent40-ioi-ema9998-s0 (n=2512): the plumbing check's target."""
TOLERANCE = 0.002
"""How far the un-fused macro may sit from `OFFICIAL` before the plumbing is distrusted."""


@dataclass(frozen=True)
class WordTags:
    """Each vocabulary word's distribution over the stacker's tags, counted in context on training rows, add-one."""

    distribution: Float[np.ndarray, "v k"]

    @classmethod
    def of(cls, targets: Int[np.ndarray, " n"], tags: Int[np.ndarray, " n"], k: int) -> "WordTags":
        """The distribution of `tags` (each below `k`, `UNKNOWN` for none) over the rows whose 50-way `targets` name a
        word and whose tag is known."""
        counts = np.ones((VOCAB_SIZE, k))
        known = (targets >= 0) & (tags != UNKNOWN)
        np.add.at(counts, (targets[known], tags[known]), 1)
        return cls(counts / counts.sum(1, keepdims=True))

    def bonus(self, posterior: Float[np.ndarray, "n k"], prior: Float[np.ndarray, " k"]) -> Float[np.ndarray, "n v"]:
        """Each row's log likelihood ratio of every word: sum_t p(t | word) p(t | MEG) / p(t)."""
        evidence = np.exp(posterior) / prior @ self.distribution.T
        return np.log(evidence + FLOOR)


@dataclass(frozen=True)
class Arm(ABC):
    """How the tag likelihood ratio narrows the vocabulary scores: a knob swept over `grid`, its first fuses nothing."""

    name: str
    grid: tuple[float, ...]

    @abstractmethod
    def fuse(self, logits: Float[Tensor, "n v"], bonus: Float[Tensor, "n v"], setting: float) -> Float[Tensor, "n v"]:
        """`logits` narrowed by the log tag likelihood `bonus` at `setting`."""


@dataclass(frozen=True)
class Hierarchical(Arm):
    """The log tag likelihood added at weight `setting`."""

    name: str = "hierarchical"
    grid: tuple[float, ...] = GRID

    @override
    def fuse(self, logits: Float[Tensor, "n v"], bonus: Float[Tensor, "n v"], setting: float) -> Float[Tensor, "n v"]:
        return logits + setting * bonus


@dataclass(frozen=True)
class Gate(Arm):
    """Every candidate whose likelihood ratio is under `setting` ranked behind the others, each by its own scores."""

    name: str = "gate"
    grid: tuple[float, ...] = THRESHOLDS

    @override
    def fuse(self, logits: Float[Tensor, "n v"], bonus: Float[Tensor, "n v"], setting: float) -> Float[Tensor, "n v"]:
        return logits - DEMOTED * (bonus.exp() < setting)


@dataclass(frozen=True)
class Domain:
    """One validation domain's unseen-span words: the vocabulary scores, each word's part-of-speech bonus, its label."""

    logits: Float[Tensor, "n v"]
    bonus: Float[Tensor, "n v"]
    labels: Int[Tensor, " n"]

    @classmethod
    def of(  # devtools-ignore: test-mirror
        cls,
        logits: Float[Tensor, "n v"],
        labels: Int[Tensor, " n"],
        words: pd.DataFrame,
        valid: Table,
        bonus: Float[np.ndarray, "m v"],
    ) -> "Domain":
        """`logits` and `labels` of `words`' rows, each given its bonus from the validation table row of the same run
        and onset; a word the table dropped for want of a tag has none."""
        rows = valid.rows(words[Column.RUN].to_numpy(), words[Column.ONSET].to_numpy(np.float64))
        found = rows >= 0
        wanted = labels.cpu().numpy()
        if (valid.targets[rows[found]] != wanted[found]).any():
            raise ValueError("a scored word and its part-of-speech row name different vocabulary words")
        aligned = np.zeros((len(rows), bonus.shape[1]))
        aligned[found] = bonus[rows[found]]
        LOG.info("%d of %d scored words carry a part of speech", found.sum(), len(rows))
        return cls(logits, torch.from_numpy(aligned).float().to(logits.device), labels)

    def fused(self, arm: "Arm", setting: float) -> Float[Tensor, "n v"]:
        """The vocabulary scores `arm` fuses with the bonus at `setting`."""
        return arm.fuse(self.logits, self.bonus, setting)


@dataclass(frozen=True)
class Grid:
    """Every domain's score at every setting of an arm, each the official ruler's per-adjustment balanced accuracy."""

    scores: dict[str, dict[float, Score]]

    @classmethod
    def of(cls, evaluator: Evaluator, log_prior: Float[Tensor, " v"], domains: dict[str, Domain], arm: Arm) -> "Grid":
        """`domains` scored at every setting of `arm`."""
        return cls(
            {
                name: {
                    setting: evaluator.evaluate(domain.fused(arm, setting), domain.labels, log_prior)
                    for setting in arm.grid
                }
                for name, domain in domains.items()
            }
        )

    @property
    def settings(self) -> tuple[float, ...]:
        return tuple(next(iter(self.scores.values())))

    def macro(self, setting: float) -> Score:
        """The macro over every domain at `setting`, the adjustment chosen once for all as the official scorer does."""
        return Score.mean(by_setting[setting] for by_setting in self.scores.values())

    def chosen(self, held: str) -> float:
        """The setting that scores best on every domain but `held`, the earlier on a tie."""
        others = [by_setting for name, by_setting in self.scores.items() if name != held]
        return max(self.settings, key=lambda setting: Score.mean(by_setting[setting] for by_setting in others).bacc10)

    def leave_one_out(self) -> tuple[Score, dict[str, float]]:
        """The macro of each domain scored at the setting the others chose, and those settings."""
        weights = {name: self.chosen(name) for name in self.scores}
        return Score.mean(self.scores[name][setting] for name, setting in weights.items()), weights

    def per_domain(self, weights: dict[str, float], macro: Score) -> dict[str, tuple[float, int]]:
        """Each domain's score at its setting, read at the macro's adjustment, and its word count."""
        return {
            name: (round(self.scores[name][setting].by_tau[macro.tau], 4), self.scores[name][setting].n)
            for name, setting in weights.items()
        }

    def report(self, arm: str, official: float) -> None:  # devtools-ignore: test-mirror
        """Log the plumbing check, the fit-on-validation grid and the leave-one-domain-out macro."""
        base = self.macro(self.settings[0])
        LOG.info("%s setting 0 macro %.4f over %d words, official %.4f", arm, base.bacc10, base.n, official)
        if abs(base.bacc10 - official) > TOLERANCE:
            raise ValueError(f"un-fused macro {base.bacc10:.4f} does not reproduce the official {official}")
        for setting in self.settings:
            macro = self.macro(setting)
            LOG.info(
                "%s fit-on-val %s macro %.4f tau %s over %d words, per domain %s",
                arm,
                setting,
                macro.bacc10,
                macro.tau,
                macro.n,
                self.per_domain(dict.fromkeys(self.scores, setting), macro),
            )
        macro, weights = self.leave_one_out()
        LOG.info(
            "%s leave-one-domain-out macro %.4f tau %s over %d words, per domain %s, settings %s",
            arm,
            macro.bacc10,
            macro.tau,
            macro.n,
            self.per_domain(weights, macro),
            weights,
        )


@dataclass(frozen=True)
class Weighting(ABC):
    """How the probe weights its classes, and the tag prior p(t) that matches it."""

    name: str
    balanced: bool

    @abstractmethod
    def prior(self, counts: Int[np.ndarray, " k"]) -> Float[np.ndarray, " k"]:
        """p(t) from the training rows' count of each tag."""

    def bonus(self, train: Table, valid: Table, stack: tuple[str, ...]) -> Float[np.ndarray, "m v"]:
        """Each validation row's log tag likelihood ratio of every word, the posterior stacked over `stack`."""
        probe = Probe(train, valid, self.balanced)
        classes, posterior = probe.stacked(stack, {name: probe.probabilities(name) for name in stack})
        tags = np.where(train.tags == UNKNOWN, UNKNOWN, np.searchsorted(classes, train.tags))
        words = WordTags.of(train.targets, tags, len(classes))
        return words.bonus(posterior, self.prior(np.bincount(tags[tags != UNKNOWN], minlength=len(classes))))

    @staticmethod
    def bonuses(train: Table, valid: Table, stack: tuple[str, ...]) -> dict[str, Float[np.ndarray, "m v"]]:
        """The bonus under every weighting, by its name."""
        return {weighting.name: weighting.bonus(train, valid, stack) for weighting in (Natural(), Balanced())}


@dataclass(frozen=True)
class Natural(Weighting):
    """Classes weighted by their frequency, the prior their share of the training tags."""

    name: str = "natural"
    balanced: bool = False

    @override
    def prior(self, counts: Int[np.ndarray, " k"]) -> Float[np.ndarray, " k"]:
        return counts / counts.sum()


@dataclass(frozen=True)
class Balanced(Weighting):
    """Classes weighted equally, the prior uniform."""

    name: str = "balanced"
    balanced: bool = True

    @override
    def prior(self, counts: Int[np.ndarray, " k"]) -> Float[np.ndarray, " k"]:
        return np.full(len(counts), 1 / len(counts))


@dataclass(frozen=True)
class Held:
    """The frozen decoder on the held-out runs: each domain's unseen-span words, as many as count."""

    evaluator: Evaluator
    model: nn.Module
    runs: dict[str, list[Run]]
    heard: pd.DataFrame
    valid: Table

    @staticmethod
    def shrunk(model: nn.Module) -> nn.Module:
        """`model` with every subject layer kept to the subjects the deep fallback reads."""
        for layer in (module for module in model.modules() if isinstance(module, SubjectLayer)):
            layer.shrink(Fallback(Unseen.DEEP).keep, Unseen.DEEP)
        return model

    def domain(self, name: str, split: Split, novel: Int[Tensor, " k"], bonus: Float[np.ndarray, "m v"]) -> Domain:
        """Domain `name`'s `novel` words of `split`, each given its row of `bonus`."""
        rows = self.heard[self.heard[Column.RUN].isin([run.name for run in self.runs[name]])].iloc[novel.cpu().numpy()]
        logits = self.evaluator.logits(self.model, split)[Head.CONTEXT][novel]
        return Domain.of(logits, split.y[novel], rows, self.valid, bonus)

    def domains(self, splits: dict[str, Split], bonus: Float[np.ndarray, "m v"]) -> dict[str, Domain]:
        """Every domain of `splits` with enough unseen-span words."""
        return {
            name: self.domain(name, split, novel, bonus)
            for name, split in splits.items()
            if len(novel := split.novel(self.evaluator.seen)) >= self.evaluator.min_novel
        }

    def report(
        self,
        splits: dict[str, Split],
        bonuses: dict[str, Float[np.ndarray, "m v"]],
        log_prior: Float[Tensor, " v"],
        official: float,
    ) -> None:
        """Log every arm's grid under every weighting's bonus."""
        for weighting, bonus in bonuses.items():
            domains = self.domains(splits, bonus)
            for arm in (Hierarchical(), Gate()):
                Grid.of(self.evaluator, log_prior, domains, arm).report(f"{weighting} {arm.name}", official)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="the decoder under models/ whose logits are narrowed")
    parser.add_argument("--embeddings", help="the model under models/ whose embeddings the probe reads (default: name)")
    parser.add_argument("--stack", choices=sorted(STACKS), default="context", help="the probes the posterior stacks")
    parser.add_argument("--official", type=float, default=OFFICIAL, help="the decoder's official macro BAcc@10")
    parser.add_argument("--stride", type=int, default=4, help="every stride-th subject-0 training run is probed on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    directories = Ensemble.trained(root.models(), args.name)
    data = DataConfig.shared(directories, DataConfig.grouping)
    reader = Reader(root, data, device, geometry=data.geometry(root, device, aligned=False))
    model = Held.shrunk(Ensemble.load([directory / "model.pt" for directory in directories], Unseen.DEEP).to(device))
    sequences, log_prior = Heard.under(root, data, device)
    evaluator = Evaluator(device, sequences, SelectionConfig.shared(directories).min_novel)
    runs = Run.discover(root.raw())
    held = reader.ruled(runs)
    trained = [run for run in data.trained(runs) if run.subject == DEEP][:: args.stride]
    contexts = ContextTags.load(ContextTags.path(root))
    timing = Timing.fitted(data.run_cache(root).table([run.name for run in trained]))
    embedded = Embedded(Checkpoint.load(root.models() / (args.embeddings or args.name) / "model.pt").to(device))
    train = Table.of(reader, embedded, {"train": trained}, contexts, timing)
    valid = Table.of(reader, embedded, held, contexts, timing)
    bonuses = Weighting.bonuses(train, valid, STACKS[args.stack])
    splits, _ = reader.domains(held)
    official = evaluator.report(model, splits, log_prior, 0, 0.0)
    LOG.info("official macro %.4f over %d words, expected %.4f", official.macro.bacc10, official.macro.n, args.official)
    _, heard = reader.load([run for group in held.values() for run in group])
    Held(evaluator, model, held, heard, valid).report(splits, bonuses, log_prior, args.official)


if __name__ == "__main__":
    main()

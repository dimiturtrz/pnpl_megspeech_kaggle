"""The adversaries: a head reading what the encoder should leave out of the representation it reads, through a
gradient reversal. The timing adversary regresses each word's log duration and sentence position, which the span
transformer already gets from the onsets (megspeech-7gf.98); the run adversary classifies the training run the word
was heard in, which a test run never shares (megspeech-prm, Ganin et al. 2016)."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import override

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.raw.runs import Column
from megspeech.objective.auxiliary.adversary.reversal import Reversal
from megspeech.objective.term import AuxiliaryTerm, Reading

HIDDEN = 128
"""Width of the adversary's hidden layer."""
TARGETS = 2
"""What the timing adversary regresses: log duration, sentence position."""
SENTENCE = [Column.RUN, Column.SENTENCE]


class Adversary(nn.Sequential):
    """A reversal, then a small MLP reading `outputs` values per word."""

    def __init__(self, width: int, outputs: int = TARGETS) -> None:
        super().__init__(Reversal(), nn.Linear(width, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, outputs))


@dataclass(frozen=True)
class Adversarial(AuxiliaryTerm):
    """An adversary's term over every training word, its labels the word's row in split order."""

    @abstractmethod
    def outputs(self) -> int:
        """Values the adversary reads per word."""

    @abstractmethod
    def rows(self) -> int:
        """Training words the term holds a target for."""

    @override
    def head(self, width: int) -> nn.Module:
        return Adversary(width, self.outputs())

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != self.rows():
            raise ValueError(f"{self.rows()} adversary rows for {len(split.y)} words")
        return torch.arange(self.rows(), device=split.y.device)


@dataclass(frozen=True)
class TimingTerm(Adversarial):
    """Squared error on `targets` (each column standardised, so predicting the mean costs 1) over the entries `known`;
    both hold every training word's row, in split order."""

    weight: float
    targets: Float[Tensor, "n 2"]
    known: Bool[Tensor, "n 2"]

    @staticmethod
    def of(weight: float, words: pd.DataFrame, device: torch.device) -> "TimingTerm":
        """The term over training `words`: log gap to the next word of the sentence (unknown for its last) and the
        word's place in the sentence, each standardised over the words it is known for."""
        start = words[Column.ONSET].to_numpy(np.float64)
        following = words.groupby(SENTENCE)[Column.ONSET].shift(-1).to_numpy(np.float64)
        position = words.groupby(SENTENCE)[Column.ONSET].rank(method="first").to_numpy(np.float64) - 1
        duration = np.log(np.maximum(following - start, np.finfo(np.float64).eps))
        raw = np.stack([duration, position], axis=1)
        known = np.stack([~np.isnan(following), np.ones(len(words), bool)], axis=1)
        mean = np.array([raw[known[:, i], i].mean() for i in range(TARGETS)])
        std = np.array([raw[known[:, i], i].std() for i in range(TARGETS)])
        standard = np.where(known, (raw - mean) / std, 0.0)
        return TimingTerm(
            weight=weight,
            targets=torch.from_numpy(standard).float().to(device),
            known=torch.from_numpy(known).to(device),
        )

    @override
    def outputs(self) -> int:
        return TARGETS

    @override
    def rows(self) -> int:
        return len(self.targets)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        target = reading.target
        at = target.clamp(min=0)
        valid = self.known[at] & (target != IGNORE)[..., None]
        error = (reading.output.float() - self.targets[at]).square()
        return self.averaged(error.flatten(), valid.flatten(), reading.rows[..., None].expand_as(valid).flatten())


@dataclass(frozen=True)
class RunTerm(Adversarial):
    """Cross-entropy of each training word's run among the `classes` training runs, `runs` holding every training
    word's run index in split order; `hits` tallies the weighted correct and seen rows on the device until drained."""

    weight: float
    runs: Int[Tensor, " n"]
    classes: int
    hits: Float[Tensor, " 2"]

    @staticmethod
    def of(weight: float, words: pd.DataFrame, device: torch.device) -> "RunTerm":
        """The term over training `words`, its classes their distinct runs in sorted order."""
        codes, runs = pd.factorize(words[Column.RUN], sort=True)
        return RunTerm(
            weight=weight,
            runs=torch.from_numpy(codes.astype(np.int64)).to(device),
            classes=len(runs),
            hits=torch.zeros(2, device=device),
        )

    @override
    def outputs(self) -> int:
        return self.classes

    @override
    def rows(self) -> int:
        return len(self.runs)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        target = reading.target.flatten()
        run = self.runs[target.clamp(min=0)]
        keep = target != IGNORE
        logits = reading.output.flatten(0, 1).float()
        rows = reading.rows.flatten()
        with torch.no_grad():
            share = keep * rows
            self.hits.add_(torch.stack([((logits.argmax(-1) == run) * share).sum(), share.sum()]))
        return self.averaged(functional.cross_entropy(logits, run, reduction="none"), keep, rows)

    @override
    def drained(self) -> dict[str, float]:
        correct, seen = self.hits.tolist()
        self.hits.zero_()
        majority = torch.bincount(self.runs, minlength=self.classes).max().item() / max(self.rows(), 1)
        return {"accuracy": correct / max(seen, 1.0), "majority": majority, "classes": float(self.classes)}


class AdversaryTarget(StrEnum):
    """What the adversary reads off each training word."""

    TIMING = "timing"
    RUN = "run"

    def term(self, words: pd.DataFrame, device: torch.device) -> Adversarial:
        """This target's term at weight 1 over training `words`: its strength is the reversal's, not a loss weight."""
        return ADVERSARIES[self](1.0, words, device)


ADVERSARIES: dict[AdversaryTarget, Callable[[float, pd.DataFrame, torch.device], Adversarial]] = {
    AdversaryTarget.TIMING: TimingTerm.of,
    AdversaryTarget.RUN: RunTerm.of,
}

"""The timing adversary: a head regressing each word's log duration and sentence position from the representation it
reads, through a gradient reversal, so the encoder is trained to leave out what the span transformer already gets
from the onsets (megspeech-7gf.98)."""

from dataclasses import dataclass
from typing import override

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.auxiliary.reversal import Reversal
from megspeech.objective.term import AuxiliaryTerm

HIDDEN = 128
"""Width of the adversary's hidden layer."""
TARGETS = 2
"""What it regresses: log duration, sentence position."""
SENTENCE = ["run", "sentence"]


class Adversary(nn.Sequential):
    """A reversal, then a small MLP regressing `TARGETS` values per word."""

    def __init__(self, width: int) -> None:
        super().__init__(Reversal(), nn.Linear(width, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, TARGETS))


@dataclass(frozen=True)
class TimingTerm(AuxiliaryTerm):
    """Squared error on `targets` (each column standardised, so predicting the mean costs 1) over the entries `known`;
    both hold every training word's row, in split order."""

    weight: float
    targets: Float[Tensor, "n 2"]
    known: Bool[Tensor, "n 2"]

    @staticmethod
    def of(weight: float, words: pd.DataFrame, device: torch.device) -> "TimingTerm":
        """The term over training `words`: log gap to the next word of the sentence (unknown for its last) and the
        word's place in the sentence, each standardised over the words it is known for."""
        start = words["onset"].to_numpy(np.float64)
        following = words.groupby(SENTENCE)["onset"].shift(-1).to_numpy(np.float64)
        position = words.groupby(SENTENCE)["onset"].rank(method="first").to_numpy(np.float64) - 1
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
    def head(self, width: int) -> nn.Module:
        return Adversary(width)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != len(self.targets):
            raise ValueError(f"{len(self.targets)} timing rows for {len(split.y)} words")
        return torch.arange(len(self.targets), device=self.targets.device)

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        rows = target.clamp(min=0)
        valid = self.known[rows] & (target != IGNORE)[..., None]
        error = (output.float() - self.targets[rows]).square() * valid
        return error.sum() / valid.sum().clamp(min=1)

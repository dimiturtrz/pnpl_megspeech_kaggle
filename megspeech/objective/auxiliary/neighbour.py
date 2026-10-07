"""The neighbour auxiliaries: each word slot's local embedding predicts the label of the word beside it in its span."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.term import AuxiliaryTerm

OUTSIDE = -1
"""A real word outside the 50: it has no label to be predicted but its own embedding still predicts its neighbour's;
`IGNORE` marks the slots with no word."""


@dataclass(frozen=True)
class NeighbourTerm(AuxiliaryTerm):
    """A 50-way head over a word's local embedding, scored against the label of its neighbour in the same span.

    The bidirectional span transformer already sees the neighbour's MEG, so the head reads the embedding before it.
    A slot's target is `IGNORE` where its neighbour is outside the 50, absent or beyond the span, and the slot's own
    word may be outside the 50.
    """

    local: ClassVar[bool] = True

    @staticmethod
    def beside(target: Int[Tensor, "g w"], offset: int) -> Int[Tensor, "g w"]:
        """Each position's entry of `target` `offset` words on in its span (negative: before), `IGNORE` past the
        span's edge."""
        length = target.shape[1]
        padded = functional.pad(target, (max(-offset, 0), max(offset, 0)), value=IGNORE)
        start = max(offset, 0)
        return padded[:, start : start + length]

    @abstractmethod
    def shift(self, target: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        """Each position's neighbour's entry of `target`, `IGNORE` past the span's edge."""

    @override
    def head(self, width: int) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, VOCAB_SIZE))

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        return torch.where(split.y == IGNORE, OUTSIDE, split.y)

    def neighbours(self, target: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        """The label to predict at each position: its neighbour's, `IGNORE` unless both words are heard and the
        neighbour is one of the 50."""
        beside = self.shift(target)
        return torch.where((target == IGNORE) | (beside < 0), IGNORE, beside)

    @override
    def weighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"], rows: Float[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        known = self.neighbours(target)
        losses = functional.cross_entropy(
            output.flatten(0, 1).float(), known.flatten(), ignore_index=IGNORE, reduction="none"
        )
        return self.averaged(losses, (known != IGNORE).flatten(), rows.flatten())


@dataclass(frozen=True)
class NextWordTerm(NeighbourTerm):
    """Each word's embedding predicts the next word's label."""

    @override
    def shift(self, target: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        return self.beside(target, 1)


@dataclass(frozen=True)
class PreviousWordTerm(NeighbourTerm):
    """Each word's embedding predicts the previous word's label."""

    @override
    def shift(self, target: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        return self.beside(target, -1)

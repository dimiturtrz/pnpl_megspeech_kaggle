"""One weighted term of a training step's loss: a readout of the model and the per-word labels it is scored against.

A step's loss is the sum of its terms: the context and word heads' (`LabelTerm`), then each auxiliary head's over
every heard word's embedding (`AuxiliaryTerm`).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.context import Objective


@dataclass(frozen=True)
class Term(ABC):
    """A readout's `weight`ed loss against the label of each span position's word."""

    weight: float = field(kw_only=True)

    def loss(
        self,
        head: nn.Module,
        output: Float[Tensor, "g w _"],
        target: Int[Tensor, "g w"],
        inside: Bool[Tensor, "g w"] | None = None,
    ) -> Float[Tensor, ""]:
        """`output` of `head` at span positions whose words carry `target`, gathered from `labels`; `inside` marks the
        positions of words in the vocabulary, which only the auxiliary terms read."""
        return self.weight * self.unweighted(head, output, target)

    @abstractmethod
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]: ...

    @abstractmethod
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        """The per-word label of `split` this term scores against."""


@dataclass(frozen=True)
class LabelTerm(Term):
    """A 50-way head trained on the vocabulary label, smoothed by `smoothing`, each class's rows scaled by
    `classes`."""

    smoothing: float
    classes: Float[Tensor, " v"] | None
    weight: float = field(default=1.0, kw_only=True)

    @staticmethod
    def balanced(y: Int[Tensor, " n"], weight: Float[Tensor, " n"], balance: float) -> Float[Tensor, " v"] | None:
        """Each class's cross-entropy weight, `count ** -balance` over the labelled `y` counted at the draw `weight`
        training samples them with, mean 1 over the classes present and 0 for the absent; None at balance 0, the
        unweighted loss itself."""
        known = y != IGNORE
        counts = torch.bincount(y[known], weight[known], minlength=VOCAB_SIZE).float()
        present = counts > 0
        weight = torch.where(present, counts.clamp(min=1).pow(-balance), 0.0)
        return weight / weight[present].mean() if balance else None

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        return split.y


@dataclass(frozen=True)
class ContextTerm(LabelTerm):
    """The context head, trained by its `objective` in float32."""

    objective: Objective = Objective.SOFTMAX

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        return split.unmuted(split.y)

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        return self.objective.loss(output.flatten(0, 1), target.flatten(), self.smoothing, self.classes)


@dataclass(frozen=True)
class DistilledTerm(ContextTerm):
    """The context head trained toward each word's label and, at `share`, toward a teacher's log-probabilities
    `taught` of the same word (Hinton et al. 2015): the KL between both distributions softened by `temperature`,
    scaled by its square so its gradient keeps the label's scale. Its labels are the split's word rows, so a batch
    gathers both the label `y` and the teacher's row."""

    taught: Float[Tensor, "n v"] = field(kw_only=True)
    y: Int[Tensor, " n"] = field(kw_only=True)
    share: float = field(kw_only=True)
    temperature: float = field(default=1.0, kw_only=True)
    unlabelled: bool = field(default=False, kw_only=True)
    """Also match the teacher on the words outside the 50, which carry no label but do carry its distribution."""

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != len(self.y):
            raise ValueError(f"{len(split.y)} words against {len(self.y)} taught rows")
        return split.unmuted(torch.arange(len(self.y), device=self.y.device))

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        rows = target.clamp(min=0)
        y = torch.where(target == IGNORE, IGNORE, self.y[rows])
        hard = super().unweighted(head, output, y)
        keep = (target != IGNORE if self.unlabelled else y != IGNORE).float()
        teacher = (self.taught[rows].float() / self.temperature).log_softmax(-1)
        student = (output.float() / self.temperature).log_softmax(-1)
        kl = (teacher.exp() * (teacher - student)).sum(-1) * self.temperature**2
        soft = (kl * keep).sum() / keep.sum().clamp(min=1)
        return (1 - self.share) * hard + self.share * soft


@dataclass(frozen=True)
class WordTerm(LabelTerm):
    """The isolated-word head's softmax at the autocast's precision: the logits stay bfloat16 into the
    cross-entropy, unlike the context head's float32, which every trained run's numbers were made with."""

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        return functional.cross_entropy(
            output.flatten(0, 1),
            target.flatten(),
            weight=self.classes,
            ignore_index=IGNORE,
            label_smoothing=self.smoothing,
        )


@dataclass(frozen=True)
class AuxiliaryTerm(Term):
    """A head over every heard word's embedding, in or out of the 50, scored against the word itself."""

    local: ClassVar[bool] = False
    """Whether the head reads the word's local embedding, before the span transformer, whatever tap the run sets."""
    timed: ClassVar[bool] = False
    """Whether the head reads the window's per-timestep encoder features (`Tap.TIME`), not a per-word vector."""
    vocab_weight: float = field(default=1.0, kw_only=True)
    """Each row of a word in the 50 weighs this against 1 for the others, in a mean normalised by the weights."""

    @staticmethod
    def averaged(values: Float[Tensor, " n"], keep: Bool[Tensor, " n"], rows: Float[Tensor, " n"]) -> Float[Tensor, ""]:
        """The mean of `values` over the `keep` rows, each at its weight in `rows`; 0 when none, without a host sync."""
        share = keep * rows
        return (values * share).sum() / share.sum().clamp(min=torch.finfo(share.dtype).tiny)

    @override
    def loss(
        self,
        head: nn.Module,
        output: Float[Tensor, "g w _"],
        target: Int[Tensor, "g w"],
        inside: Bool[Tensor, "g w"] | None = None,
    ) -> Float[Tensor, ""]:
        rows = torch.ones(target.shape, device=target.device)
        if inside is not None:
            rows = rows + (self.vocab_weight - 1.0) * inside.float()
        return self.weight * self.weighted(head, output, target, rows)

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        return self.weighted(head, output, target, torch.ones(target.shape, device=target.device))

    @abstractmethod
    def weighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"], rows: Float[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        """The loss with each position's row at its weight in `rows`."""

    @abstractmethod
    def head(self, width: int) -> nn.Module: ...

    def readout(self, word_head: nn.Module, width: int) -> nn.Module:
        """The head this term trains over `width`-wide representations: its own, unless it reads one the model's
        `word_head` already holds."""
        return self.head(width)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        return split.token


class Classified(AuxiliaryTerm):
    """Each heard word classified into `count` classes, the words without a class (`UNKNOWN`) left out."""

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def classify(self, token: Int[Tensor, "g w"]) -> Int[Tensor, "g w *p"]: ...

    @staticmethod
    def masked(
        logits: Float[Tensor, "n k"], classes: Int[Tensor, " n"], rows: Float[Tensor, " n"]
    ) -> Float[Tensor, ""]:
        """Cross-entropy over the rows with a known class, each at its weight in `rows`; 0 when there are none,
        without a host sync."""
        losses = functional.cross_entropy(logits.float(), classes.clamp(min=0), reduction="none")
        return AuxiliaryTerm.averaged(losses, classes != UNKNOWN, rows)

    @override
    def weighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"], rows: Float[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        classes = self.classify(target)
        spread = rows.reshape(*rows.shape, *[1] * (classes.ndim - rows.ndim)).expand_as(classes)
        return self.masked(output.reshape(-1, self.count()), classes.flatten(), spread.flatten())

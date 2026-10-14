"""One weighted term of a training step's loss: a readout of the model and the per-word labels it is scored against.

A step's loss is the sum of its terms: the context and word heads' (`LabelTerm`), then each auxiliary head's over
every heard word's embedding (`AuxiliaryTerm`).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar, override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.context import Objective
from megspeech.objective.topk import TopKMix


@dataclass(frozen=True)
class Reading:
    """A readout `head`'s `output` at every span position, against the `target` its word carries, each position's
    row at its weight in `rows`: what a term scores."""

    head: nn.Module
    output: Float[Tensor, "g w _"]
    target: Int[Tensor, "g w"]
    rows: Float[Tensor, "g w"]

    def contrasted(
        self,
        contrast: Callable[
            [Float[Tensor, "n d"], Float[Tensor, "n e"], Int[Tensor, " n"], Float[Tensor, " n"]], Float[Tensor, ""]
        ],
        table: Float[Tensor, "v e"],
        kinds: Int[Tensor, "g w"],
    ) -> Float[Tensor, ""]:
        """`contrast` of every position's output against the `table` row its word `kinds` names, a negative kind
        naming none."""
        flat = kinds.flatten()
        return contrast(self.output.flatten(0, 1), table[flat.clamp(min=0)], flat, self.rows.flatten())


@dataclass(frozen=True)
class Term(ABC):
    """A readout's `weight`ed loss against the label of each span position's word."""

    weight: float = field(kw_only=True)

    def read(
        self,
        head: nn.Module,
        output: Float[Tensor, "g w _"],
        target: Int[Tensor, "g w"],
        inside: Bool[Tensor, "g w"] | None = None,
    ) -> Reading:
        """`output` of `head` at span positions whose words carry `target`, gathered from `labels`, every row at
        weight 1; `inside` marks the positions of words in the vocabulary, which only the auxiliary terms weigh."""
        return Reading(head, output, target, torch.ones(target.shape, device=target.device))

    def loss(self, reading: Reading) -> Float[Tensor, ""]:
        """The `weight`ed loss of `reading`."""
        return self.weight * self.unweighted(reading)

    def drained(self) -> dict[str, float]:
        """What the term tallied over the steps since the last drain, then emptied; nothing for a term that tallies
        nothing."""
        return {}

    @abstractmethod
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        """The loss of `reading` before the term's weight."""

    @abstractmethod
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        """The per-word label of `split` this term scores against."""


@dataclass(frozen=True)
class LabelTerm(Term):
    """A 50-way head trained on the vocabulary label, smoothed by `smoothing`, each class's rows scaled by
    `classes`, its loss blended with the smooth top-k term by `topk`."""

    smoothing: float
    classes: Float[Tensor, " v"] | None
    weight: float = field(default=1.0, kw_only=True)
    topk: TopKMix = field(default=TopKMix(), kw_only=True)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        logits, y = reading.output.flatten(0, 1), reading.target.flatten()
        return self.topk.mixed(self.labelled(logits, y), logits, y, self.classes)

    @abstractmethod
    def labelled(self, logits: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> Float[Tensor, ""]:
        """The head's own loss over `logits` against `y`, before the top-k blend."""

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

    def hard(self, target: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        """The vocabulary label of each position of `target`, `IGNORE` for words outside the 50."""
        return target

    @override
    def labelled(self, logits: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> Float[Tensor, ""]:
        return self.objective.loss(logits, y, self.smoothing, self.classes)


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
    def hard(self, target: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        return torch.where(target == IGNORE, IGNORE, self.y[target.clamp(min=0)])

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        target = reading.target
        rows = target.clamp(min=0)
        y = self.hard(target)
        hard = super().unweighted(replace(reading, target=y))
        keep = (target != IGNORE if self.unlabelled else y != IGNORE).float()
        teacher = (self.taught[rows].float() / self.temperature).log_softmax(-1)
        student = (reading.output.float() / self.temperature).log_softmax(-1)
        kl = (teacher.exp() * (teacher - student)).sum(-1) * self.temperature**2
        soft = (kl * keep).sum() / keep.sum().clamp(min=1)
        return (1 - self.share) * hard + self.share * soft


@dataclass(frozen=True)
class WordTerm(LabelTerm):
    """The isolated-word head's softmax at the autocast's precision: the logits stay bfloat16 into the
    cross-entropy, unlike the context head's float32, which every trained run's numbers were made with."""

    @override
    def labelled(self, logits: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> Float[Tensor, ""]:
        return functional.cross_entropy(
            logits,
            y,
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
        """The mean of `values` over the `keep` rows, each at its weight in `rows`, without a host sync; 0 when none.

        A batch can hold no row a term covers (its targets are sparse: words without audio, outside the lexicon or
        unknown), and then adds no gradient rather than a NaN. A term covering no training word at all is refused
        where its table is read (`Speech.covering`), so a zero here is never a dead term."""
        share = keep * rows
        return (values * share).sum() / share.sum().clamp(min=torch.finfo(share.dtype).tiny)

    def emphasis(self, target: Int[Tensor, "g w"], inside: Bool[Tensor, "g w"] | None = None) -> Float[Tensor, "g w"]:
        """Each position's row weight: `vocab_weight` for the words `inside` the 50, 1 for every other, all 1 when
        nothing marks them."""
        rows = torch.ones(target.shape, device=target.device)
        if inside is not None:
            rows = rows + (self.vocab_weight - 1.0) * inside.float()
        return rows

    @override
    def read(
        self,
        head: nn.Module,
        output: Float[Tensor, "g w _"],
        target: Int[Tensor, "g w"],
        inside: Bool[Tensor, "g w"] | None = None,
    ) -> Reading:
        return Reading(head, output, target, self.emphasis(target, inside))

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
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        classes = self.classify(reading.target)
        rows = reading.rows
        spread = rows.reshape(*rows.shape, *[1] * (classes.ndim - rows.ndim)).expand_as(classes)
        return self.masked(reading.output.reshape(-1, self.count()), classes.flatten(), spread.flatten())

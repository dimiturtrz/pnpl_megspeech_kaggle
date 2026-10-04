"""How training draws spans as it goes: their natural mix, annealed toward the Deep subject, or tilted toward the
strata training fits worst."""

from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.raw.runs import DEEP
from megspeech.data.spans.spans import PAD

BINS = 3
"""Quantile bins of span length and of word rate within a stratum: terciles, so each bin holds a third of the spans
and a subject's strata stay large enough to measure an epoch's loss on."""


class Curriculum(Protocol):
    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        """Each span's draw weight in `epoch` of `epochs`."""
        ...

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Training's context-head `logits` on the words of `spans`, for a curriculum that draws by its loss."""
        ...


class Natural:
    """Every span at its own weight, whichever subject heard it: the mix the corpus holds."""

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        return train.weight[train.index[:, 0]]

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Nothing: the draw ignores the loss."""


@dataclass(frozen=True)
class Annealed:
    """Subjects drawn equally at the first epoch, the Deep subject's share then rising linearly to `share` at the
    last; the others split the rest equally, and within a subject each span keeps its own weight.

    Many subjects early regularise the shared encoder toward what every head has in common; the Deep subject late
    fits it to the head it is scored on.
    """

    share: float

    @classmethod
    def of(cls, share: float) -> Curriculum:
        """Annealed toward `share`; the natural mix at 0."""
        return cls(share) if share else Natural()

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        first = train.index[:, 0]
        weight, subject = train.weight[first], train.subject[first]
        present, which = torch.unique(subject, return_inverse=True)
        totals = torch.zeros(len(present), dtype=weight.dtype, device=weight.device).index_add_(0, which, weight)
        progress = epoch / max(epochs - 1, 1)
        deep = (1 - progress) / len(present) + progress * self.share
        others = (1 - deep) / max(len(present) - 1, 1)
        shares = torch.where(present == DEEP, deep, others)
        return weight / totals[which] * shares[which]

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Nothing: the schedule ignores the loss."""


@dataclass
class Robust:
    """`base`'s draw tilted toward the strata training fits worst: each stratum's spans weighted by its mean word loss
    over the last epoch against every word's, to the `power`, never below `base`'s own weight (group DRO, Sagawa et
    al. 2020, as a sampler rather than a loss: balanced data, not a reshaped objective). A stratum is a subject, a
    span-length bin and a word-rate bin (`stratify`); the loss is training's, so validation never steers the draw.
    """

    base: Curriculum
    strata: Int[Tensor, " g"]
    power: float
    tilt: Float[Tensor, " s"]
    sums: Float[Tensor, " s"]
    counts: Float[Tensor, " s"]

    @classmethod
    def of(cls, base: Curriculum, train: Split, power: float) -> Curriculum:
        """`base` tilted at `power` over `train`'s strata; `base` itself at 0."""
        if not power:
            return base
        strata = cls.stratify(train)
        ones = torch.ones(int(strata.max()) + 1, device=strata.device)
        return cls(base, strata, power, ones, torch.zeros_like(ones), torch.zeros_like(ones))

    @staticmethod
    def stratify(train: Split) -> Int[Tensor, " g"]:
        """Each span's stratum: its subject, then its word count's and its words per second's `BINS`-quantile bins
        (a one-word span's rate is 0)."""
        length = (train.index != PAD).sum(dim=1)
        last = train.onset.gather(1, (length - 1).clamp(min=0)[:, None])[:, 0]
        rate = torch.where(length > 1, (length - 1) / (last - train.onset[:, 0]), 0.0)
        subject = train.subject[train.index[:, 0]]
        return (subject * BINS + Robust.binned(length.float())) * BINS + Robust.binned(rate)

    @staticmethod
    def binned(values: Float[Tensor, " g"]) -> Int[Tensor, " g"]:
        """Each value's `BINS`-quantile bin, 0 the lowest."""
        edges = torch.quantile(values, torch.linspace(0, 1, BINS + 1, device=values.device)[1:-1])
        return torch.bucketize(values, edges)

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Each labelled word's cross-entropy summed into its span's stratum, without a host sync."""
        words = functional.cross_entropy(
            logits.detach().flatten(0, 1).float(), target.flatten(), ignore_index=IGNORE, reduction="none"
        )
        stratum = self.strata[spans]
        self.sums.index_add_(0, stratum, words.view_as(target).sum(dim=1))
        self.counts.index_add_(0, stratum, (target != IGNORE).sum(dim=1).float())

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        """`base`'s weights times each span's stratum tilt, refreshed from the losses observed since the last draw;
        a stratum unseen since keeps its tilt."""
        mean = self.sums / self.counts.clamp(min=1)
        overall = self.sums.sum() / self.counts.sum()
        self.tilt = torch.where(self.counts > 0, (mean / overall).clamp(min=1).pow(self.power), self.tilt)
        self.sums.zero_()
        self.counts.zero_()
        return self.base.weights(train, epoch, epochs) * self.tilt[self.strata]

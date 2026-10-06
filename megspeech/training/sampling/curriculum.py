"""How training draws spans as it goes: their natural mix, annealed toward the Deep subject, or tilted toward the
strata training fits worst, or prioritised span by span by how rarely drawn and how hard."""

from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.raw.runs import DEEP
from megspeech.data.spans.spans import PAD

BINS = 3
"""Quantile bins of span length and of word rate within a stratum: terciles, so each bin holds a third of the spans
and a subject's strata stay large enough to measure an epoch's loss on."""

FLOOR = 0.25
"""Share of its base weight no prioritised span falls below, so easy or often-seen spans are never starved."""

CEILING = 4.0
"""Multiple of its base weight no prioritised span rises above, so no span dominates the draw."""


class Curriculum(Protocol):
    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        """Each span's draw weight in `epoch` of `epochs`."""
        ...

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Training's context-head `logits` on the words of `spans`, for a curriculum that draws by its loss."""
        ...

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """How `weights`, this epoch's draw, sit against the curriculum's own: empty where nothing tilts them."""
        ...


class Natural:
    """Every span at its own weight, whichever subject heard it: the mix the corpus holds."""

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        return train.weight[train.index[:, 0]]

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Nothing: the draw ignores the loss."""

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """Nothing to report: the draw is not tilted by a loss."""
        return {}


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

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """Nothing to report: the draw is not tilted by a loss."""
        return {}


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

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """Nothing to report: the stratum tilts are already logged by their effect on the loss."""
        return {}


@dataclass
class Prioritised:
    """`base`'s draw reweighted span by span: `(1 + seen) ** -visibility` for how often training has drawn it, times
    its difficulty (the EMA of its mean word cross-entropy) over the mean difficulty, to the `power`; never below
    `FLOOR` nor above `CEILING` times its `base` weight (a negative `power` is self-paced: it down-weights the
    hard), and a `capped` span never above it. An unseen span takes the highest difficulty so far, as prioritised
    replay gives a new transition the maximum priority (Schaul et al. 2016); a seen one is tilted by its training
    loss as in online batch selection (Loshchilov & Hutter 2015), so validation never steers the draw.
    """

    base: Curriculum
    power: float
    visibility: float
    seen: Float[Tensor, " g"]
    difficulty: Float[Tensor, " g"]
    capped: Bool[Tensor, " g"]

    @classmethod
    def of(
        cls, base: Curriculum, train: Split, power: float, visibility: float, capped: Bool[Tensor, " g"] | None = None
    ) -> Curriculum:
        """`base` prioritised over `train`'s spans; `base` itself when `power` and `visibility` are both 0."""
        if not (power or visibility):
            return base
        zeros = torch.zeros(train.index.shape[0], device=train.index.device)
        return cls(base, power, visibility, zeros, zeros.clone(), zeros.bool() if capped is None else capped)

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Each drawn span counted, and its difficulty moved halfway to this batch's mean labelled-word loss, without a
        host sync; a span with no labelled word keeps its difficulty."""
        words = functional.cross_entropy(
            logits.detach().flatten(0, 1).float(), target.flatten(), ignore_index=IGNORE, reduction="none"
        )
        counts = (target != IGNORE).sum(dim=1).float()
        loss = words.view_as(target).sum(dim=1) / counts.clamp(min=1)
        before = self.difficulty[spans]
        moved = torch.where(before > 0, (before + loss) / 2, loss)
        self.difficulty[spans] = torch.where(counts > 0, moved, before)
        self.seen.index_add_(0, spans, torch.ones_like(loss))

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        """`base`'s weights scaled by each span's visibility and difficulty, floored and capped against them."""
        base = self.base.weights(train, epoch, epochs)
        filled = torch.where(self.difficulty > 0, self.difficulty, self.difficulty.max())
        mean = filled.mean()
        relative = torch.where(mean > 0, filled / mean.clamp(min=torch.finfo(mean.dtype).tiny), torch.ones_like(filled))
        scaled = base * (1 + self.seen).pow(-self.visibility) * relative.pow(self.power)
        bounded = torch.minimum(torch.maximum(scaled, FLOOR * base), CEILING * base)
        return torch.where(self.capped, torch.minimum(bounded, base), bounded)

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """`weights` over `base`'s: the 5th, 50th and 95th percentile of the ratio, the share of spans on `FLOOR` and on
        `CEILING` (a tilt saturating its clamps), and the effective sample size over the span count."""
        ratio = weights / self.base.weights(train, epoch, epochs)
        low, mid, high = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], device=ratio.device)).tolist()
        total = weights.double()
        return {
            "ratio_p05": low,
            "ratio_p50": mid,
            "ratio_p95": high,
            "at_floor": ratio.isclose(torch.tensor(FLOOR, device=ratio.device)).float().mean().item(),
            "at_ceiling": ratio.isclose(torch.tensor(CEILING, device=ratio.device)).float().mean().item(),
            "ess": (total.sum() ** 2 / total.square().sum() / len(weights)).item(),
        }

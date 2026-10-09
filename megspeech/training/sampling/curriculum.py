"""How training draws spans as it goes: a base distribution (uniform, the corpus's natural mix, or balanced by
stratum) that modifiers wrap, each forwarding what training observes to what it wraps: annealed toward the Deep
subject, tilted toward the strata training fits worst, prioritised span by span by how rarely drawn and how hard."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.raw.runs import DEEP
from megspeech.data.spans.spans import PAD
from megspeech.training.sampling.strata import Strata

FLOOR = 0.25
"""Share of its base weight no prioritised span falls below, so easy or often-seen spans are never starved."""

CEILING = 4.0
"""Multiple of its base weight no prioritised span rises above, so no span dominates the draw."""

QUARTILE = 0.25
"""Share of the labels training holds, rarest first, whose spans `Balanced` reports the draw's share on."""


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


@dataclass(frozen=True)
class Uniform:
    """Every span at weight 1, whichever run or subject heard it."""

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        return torch.ones(len(train.index), device=train.index.device)

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Nothing: the draw ignores the loss."""

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """Nothing to report: the draw is not tilted by a loss."""
        return {}


@dataclass(frozen=True)
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
    base: Curriculum

    @classmethod
    def of(cls, share: float, base: Curriculum) -> Curriculum:
        """`base` annealed toward `share`; `base` itself at 0."""
        return cls(share, base) if share else base

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        weight, subject = self.base.weights(train, epoch, epochs), train.subject[train.index[:, 0]]
        present, which = torch.unique(subject, return_inverse=True)
        totals = torch.zeros(len(present), dtype=weight.dtype, device=weight.device).index_add_(0, which, weight)
        progress = epoch / max(epochs - 1, 1)
        deep = (1 - progress) / len(present) + progress * self.share
        others = (1 - deep) / max(len(present) - 1, 1)
        shares = torch.where(present == DEEP, deep, others)
        return weight / totals[which] * shares[which]

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Forwarded to `base`: the schedule ignores the loss."""
        self.base.observe(spans, logits, target)

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """Nothing to report: the draw is not tilted by a loss."""
        return {}


@dataclass(frozen=True)
class Stratified:
    """Each span at `count ** -alpha` of its stratum's span count (`Strata.sourced`: source, subject, span length and
    word rate), so at 1 every stratum draws equally whatever its size and at 0 every span draws alike: balanced
    data rather than a reshaped objective, as inverse-frequency sampling of rare groups (Sagawa et al. 2020)."""

    alpha: float
    strata: Int[Tensor, " g"]

    @classmethod
    def of(cls, alpha: float, train: Split) -> "Stratified":
        """The strata of `train` at exponent `alpha`."""
        return cls(alpha, Strata.sourced(train))

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        counts = torch.bincount(self.strata).float()
        return counts[self.strata].pow(-self.alpha)

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Nothing: the draw ignores the loss."""

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """Nothing to report: the draw is not tilted by a loss."""
        return {}


@dataclass(frozen=True)
class Balanced:
    """`base`'s draw tilted toward spans of rare target words: each span's weight times the mean, over its labelled
    words, of `share ** -power` for each word's label's share of training's labelled words (normalised to a mean of 1
    over those words; a span with none keeps 1). Rare words are seen more often from balanced data, not a reweighted
    loss: inverse-frequency sampling as Mikolov et al. 2013 subsample frequent words, at a tunable `power`."""

    base: Curriculum
    rarity: Float[Tensor, " g"]
    rare: Bool[Tensor, " g"]

    @classmethod
    def of(cls, base: Curriculum, train: Split, power: float) -> Curriculum:
        """`base` balanced at `power` over `train`'s labels; `base` itself at 0."""
        if not power:
            return base
        labels, valid = cls.labels(train)
        counts = torch.bincount(labels[valid], minlength=VOCAB_SIZE).float()
        word = (counts.clamp(min=1) / counts.sum()).pow(-power)
        word = word / word[labels[valid]].mean()
        spans = valid.sum(dim=1)
        summed = torch.where(valid, word[labels], 0).sum(dim=1)
        rarity = torch.where(spans > 0, summed / spans.clamp(min=1), 1.0)
        present = counts > 0
        ranked = torch.where(present, counts, torch.inf).argsort()
        rarest = ranked[: max(1, int(int(present.sum()) * QUARTILE))]
        rare = (torch.isin(labels, rarest) & valid).any(dim=1)
        return cls(base, rarity, rare)

    @staticmethod
    def labels(train: Split) -> tuple[Int[Tensor, "g w"], Bool[Tensor, "g w"]]:
        """Each span slot's label (0 where padded or unlabelled) and whether it is a labelled word."""
        labels = train.y[train.index.clamp(min=0)]
        valid = (train.index != PAD) & (labels != IGNORE)
        return torch.where(valid, labels, 0), valid

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        return self.base.weights(train, epoch, epochs) * self.rarity

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Forwarded to `base`: the tilt ignores the loss."""
        self.base.observe(spans, logits, target)

    def summary(self, train: Split, epoch: int, epochs: int, weights: Float[Tensor, " g"]) -> dict[str, float]:
        """The draw's share on spans holding a word of the rarest `QUARTILE` of labels, under `weights` and `base`'s."""
        base = self.base.weights(train, epoch, epochs)
        return {"rare_share": self.share(weights), "rare_share_base": self.share(base)}

    def share(self, weights: Float[Tensor, " g"]) -> float:
        """`weights`' share on spans holding a word of the rarest `QUARTILE` of labels."""
        return (weights[self.rare].sum() / weights.sum()).item()


class Base(StrEnum):
    """The distribution over spans a sampler's modifiers start from."""

    UNIFORM = "uniform"
    NATURAL = "natural"
    STRATIFIED = "stratified"

    def of(self, train: Split, alpha: float) -> Curriculum:
        """This base over `train`'s spans; `alpha` is `Stratified`'s exponent."""
        builders = {
            Base.UNIFORM: Uniform,
            Base.NATURAL: Natural,
            Base.STRATIFIED: lambda: Stratified.of(alpha, train),
        }
        return builders[self]()


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
        strata = Strata.stratify(train)
        ones = torch.ones(int(strata.max()) + 1, device=strata.device)
        return cls(base, strata, power, ones, torch.zeros_like(ones), torch.zeros_like(ones))

    def observe(self, spans: Int[Tensor, " b"], logits: Float[Tensor, "b w v"], target: Int[Tensor, "b w"]) -> None:
        """Each labelled word's cross-entropy summed into its span's stratum, without a host sync, and `base` told."""
        self.base.observe(spans, logits, target)
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
        host sync; a span with no labelled word keeps its difficulty. `base` is told too."""
        self.base.observe(spans, logits, target)
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

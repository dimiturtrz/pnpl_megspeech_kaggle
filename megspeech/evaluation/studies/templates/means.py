"""Per-word mean windows of the templates study: channel scaling, streamed (split-half) sums, the templates and
their split-half reliability."""

from dataclasses import dataclass
from typing import Self

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE

FLOOR = 1e-6
SPLIT_HALF = 2


@dataclass
class Scale:
    """Per-channel centring and scaling, from the training windows' own moments."""

    mean: Float[Tensor, " c"]
    std: Float[Tensor, " c"]

    def apply(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c t"]:
        return (x - self.mean[:, None]) / self.std[:, None]

    def means(self, sums: Float[Tensor, "... c t"], counts: Float[Tensor, "..."]) -> Float[Tensor, "... c t"]:
        """Scaled means of `sums` over `counts` windows (zero where a cell holds none)."""
        mean = sums / counts.clamp(min=1)[..., None, None]
        return torch.where(counts[..., None, None] > 0, (mean - self.mean[:, None]) / self.std[:, None], 0.0)


@dataclass(frozen=True)
class Templates:
    """Scaled mean window per (group, word) and the trials behind each; a group is a dataset, or all of them."""

    means: Float[Tensor, "g v c t"]
    counts: Int[Tensor, "g v"]

    @staticmethod
    def standardised(flat: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """`flat` centred and scaled to unit norm along its last axis: a dot product of two is their correlation."""
        return nn.functional.normalize(flat - flat.mean(dim=-1, keepdim=True), dim=-1)

    def pooled(self) -> "Templates":
        """One group: every dataset's mean weighted by its trials."""
        weight = self.counts.float()[..., None, None]
        total = self.counts.sum(dim=0)
        mean = (self.means * weight).sum(dim=0) / total.clamp(min=1).float()[:, None, None]
        return Templates(mean[None], total[None])

    def distance(self, group: int = 0) -> Float[Tensor, "v v"]:
        """1 - the correlation of every pair of words' flattened templates in `group`."""
        flat = Templates.standardised(self.means[group].flatten(1))
        return 1 - flat @ flat.T


@dataclass
class Sums:
    """Streamed per-(dataset, word) window sums, with the same sums over `reps` random halves of each word's trials,
    and the per-channel moments of every window."""

    sums: Float[Tensor, "g v c t"]
    counts: Float[Tensor, "g v"]
    halves: Float[Tensor, "r g h c t"]
    half_counts: Float[Tensor, "r g h"]
    moments: Float[Tensor, "m c"]
    seen: float
    generator: torch.Generator

    @classmethod
    def empty(cls, groups: int, reps: int, shape: tuple[int, int], device: torch.device, seed: int = 0) -> Self:
        channels, samples = shape
        return cls(
            torch.zeros(groups, VOCAB_SIZE, channels, samples, device=device),
            torch.zeros(groups, VOCAB_SIZE, device=device),
            torch.zeros(reps, groups, SPLIT_HALF * VOCAB_SIZE, channels, samples, device=device),
            torch.zeros(reps, groups, SPLIT_HALF * VOCAB_SIZE, device=device),
            torch.zeros(2, channels, device=device, dtype=torch.float64),
            0.0,
            torch.Generator(device=device).manual_seed(seed),
        )

    def add(self, group: int, x: Float[Tensor, "b c t"], y: Int[Tensor, " b"]) -> None:
        """Count the labelled windows `x` of dataset `group` in."""
        self.sums[group].index_add_(0, y, x)
        self.counts[group] += torch.bincount(y, minlength=VOCAB_SIZE)
        dealt = torch.randint(0, SPLIT_HALF, (len(y), len(self.halves)), generator=self.generator, device=y.device)
        for rep in range(len(self.halves)):
            cell = dealt[:, rep] * VOCAB_SIZE + y
            self.halves[rep, group].index_add_(0, cell, x)
            self.half_counts[rep, group] += torch.bincount(cell, minlength=SPLIT_HALF * VOCAB_SIZE)
        self.moments[0] += x.sum(dim=(0, 2)).double()
        self.moments[1] += x.pow(2).sum(dim=(0, 2)).double()
        self.seen += len(y) * x.shape[2]

    def scale(self) -> Scale:
        mean = self.moments[0] / self.seen
        std = (self.moments[1] / self.seen - mean**2).clamp(min=FLOOR).sqrt()
        return Scale(mean.float(), std.float())

    def templates(self, scale: Scale) -> Templates:
        return Templates(scale.means(self.sums, self.counts), self.counts.long())

    def split(
        self, scale: Scale, group: int | None
    ) -> tuple[Float[Tensor, "r v c t"], Float[Tensor, "r v c t"], Bool[Tensor, "r v"]]:
        """The two half-templates of every word and repeat in `group` (every dataset when None), and where both
        halves hold a trial."""
        halves = self.halves.sum(dim=1) if group is None else self.halves[:, group]
        counts = self.half_counts.sum(dim=1) if group is None else self.half_counts[:, group]
        means = scale.means(halves, counts)
        return means[:, :VOCAB_SIZE], means[:, VOCAB_SIZE:], (counts[:, :VOCAB_SIZE] > 0) & (counts[:, VOCAB_SIZE:] > 0)


@dataclass(frozen=True)
class Reliability:
    """Half-template correlations: each word's with its own other half, and its mean with the other words'."""

    same: Float[Tensor, "r v"]
    diff: Float[Tensor, "r v"]

    @classmethod
    def of(cls, a: Float[Tensor, "r v c t"], b: Float[Tensor, "r v c t"], valid: Bool[Tensor, "r v"]) -> "Reliability":
        """NaN for a word a half of which holds no trial."""
        first, second = Templates.standardised(a.flatten(2)), Templates.standardised(b.flatten(2))
        corr = first @ second.transpose(1, 2)
        pair = valid[:, :, None] & valid[:, None, :] & ~torch.eye(VOCAB_SIZE, dtype=torch.bool, device=a.device)
        others = pair.sum(dim=2)
        diff = (corr * pair).sum(dim=2) / others.clamp(min=1)
        nan = torch.tensor(float("nan"), device=a.device)
        return cls(
            torch.where(valid, corr.diagonal(dim1=1, dim2=2), nan),
            torch.where(valid & (others > 0), diff, nan),
        )

    def spearman_brown(self) -> Float[Tensor, " v"]:
        """Each word's full-template reliability `2r / (1 + r)` from its mean half correlation."""
        half = self.same.nanmean(dim=0)
        return 2 * half / (1 + half)

    def summary(self, counts: Int[Tensor, " v"]) -> str:
        """Mean same / different correlation, the Spearman-Brown reliabilities and the trials per word."""
        brown = self.spearman_brown()
        held = counts[counts > 0].float()
        return (
            f"words={int((~brown.isnan()).sum())} trials/word min={int(held.min())} median={held.median():.0f} "
            f"max={int(held.max())} same={self.same.nanmean():.4f} diff={self.diff.nanmean():.4f} "
            f"SB mean={brown.nanmean():.4f} median={brown.nanmedian():.4f} min={brown[~brown.isnan()].min():.4f} "
            f"max={brown[~brown.isnan()].max():.4f}"
        )

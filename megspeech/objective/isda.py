"""Implicit semantic data augmentation of the word head (Wang et al., NeurIPS 2019).

Each training feature `f` the head's linear layer reads is augmented, in expectation and in closed form, along its
class's own directions of variation `f ~ N(f, lambda * Sigma_y)`: the cross-entropy's upper bound over that Gaussian is
the cross-entropy of the logits `z_j + (lambda / 2) (w_j - w_y)^T Sigma_y (w_j - w_y)`. `Sigma_y` is a buffer outside
the model, each class's covariance of `f` over every training batch seen so far, the batches merged at their counts
(the paper's online estimate, not an EMA); it carries no gradient, the head's weights `w` do. Evaluation never reads
it: the model's own logits are untouched, only the training loss sees the augmented ones.
"""

from dataclasses import dataclass, field
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE


class Unaugmented:
    """The word head's logits as the model read them."""

    def augmented(self, logits: Float[Tensor, "g w v"], target: Int[Tensor, "g w"]) -> Float[Tensor, "g w v"]:
        """The logits the training loss reads against `target`."""
        return logits

    def ramp(self, epoch: int, epochs: int) -> float:
        """Set the strength `epoch` of `epochs` trains at, and return it."""
        return 0.0

    def drained(self) -> float:
        """The strength the epoch trained at."""
        return 0.0


@dataclass(eq=False)
class Isda(Unaugmented):
    """ISDA over the features `linear` reads, its strength raised linearly from 0 to `strength` over the run."""

    strength: float
    linear: nn.Linear
    scale: float = field(default=0.0, init=False)
    features: Tensor | None = field(default=None, init=False)
    mean: Float[Tensor, "v d"] = field(init=False)
    covariance: Float[Tensor, "v d d"] = field(init=False)
    count: Float[Tensor, " v"] = field(init=False)

    def __post_init__(self) -> None:
        weight = self.linear.weight
        width = self.linear.in_features
        self.mean = torch.zeros(VOCAB_SIZE, width, device=weight.device)
        self.covariance = torch.zeros(VOCAB_SIZE, width, width, device=weight.device)
        self.count = torch.zeros(VOCAB_SIZE, device=weight.device)
        self.linear.register_forward_pre_hook(self._seen)

    @staticmethod
    def of(strength: float, head: nn.Module) -> Unaugmented:
        """ISDA at `strength` over the word `head`'s last linear layer; nothing at strength 0."""
        if not strength:
            return Unaugmented()
        last = head[-1] if isinstance(head, nn.Sequential) else head
        if not isinstance(last, nn.Linear):
            raise TypeError(f"ISDA augments a linear word head's features, not {type(head).__name__}'s")
        return Isda(strength, last)

    def _seen(self, module: nn.Module, args: tuple[Tensor, ...]) -> None:
        """Keep the training features the linear layer is about to read."""
        if module.training:
            self.features = args[0].detach()

    @override
    def ramp(self, epoch: int, epochs: int) -> float:
        self.scale = self.strength * epoch / epochs
        return self.scale

    @override
    def drained(self) -> float:
        return self.scale

    @override
    def augmented(self, logits: Float[Tensor, "g w v"], target: Int[Tensor, "g w"]) -> Float[Tensor, "g w v"]:
        if self.features is None:
            raise RuntimeError("ISDA saw no training features: the linear layer has not run in training mode")
        y = target.flatten()
        with torch.autocast(logits.device.type, enabled=False):
            self.update(self.features.flatten(0, -2).float(), y)
            weight = self.linear.weight
            spread = self.spread(weight.float())
        added = spread[y.clamp(min=0)] * (y != IGNORE).float()[:, None]
        return logits + (self.scale / 2) * added.reshape(logits.shape)

    @torch.no_grad()
    def update(self, features: Float[Tensor, "n d"], y: Int[Tensor, " n"]) -> None:
        """Merge the batch's per-class mean and covariance of `features` into the running ones at their counts; rows
        labelled `IGNORE` count toward no class."""
        rows = functional.one_hot(y.clamp(min=0), VOCAB_SIZE).float() * (y != IGNORE).float()[:, None]
        counts = rows.sum(0)
        mean = rows.T @ features / counts.clamp(min=1)[:, None]
        centred = (features - mean[y.clamp(min=0)]) * rows.sum(1, keepdim=True)
        covariance = torch.einsum("nc,nd,ne->cde", rows, centred, centred) / counts.clamp(min=1)[:, None, None]
        share = counts / (self.count + counts).clamp(min=1)
        gap = self.mean - mean
        kept = share[:, None, None]
        self.covariance.mul_(1 - kept).add_(kept * covariance + kept * (1 - kept) * gap[:, :, None] * gap[:, None, :])
        self.mean.mul_(1 - share[:, None]).add_(share[:, None] * mean)
        self.count.add_(counts)

    def spread(self, weight: Float[Tensor, "v d"]) -> Float[Tensor, "v v"]:
        """`(w_j - w_c)^T Sigma_c (w_j - w_c)` for each true class `c` (row) and candidate `j` (column)."""
        gap = weight[None, :, :] - weight[:, None, :]
        return torch.einsum("cjd,cde,cje->cj", gap, self.covariance, gap)

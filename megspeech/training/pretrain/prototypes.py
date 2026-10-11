"""Prototype heads for the self-distillation objectives (DINO, Caron et al. 2021; iBOT, Zhou et al. 2022; DINOv2,
Oquab et al. 2023): a feature is scored against K learned prototypes, and a student matches the distribution its
EMA teacher puts over them."""

from dataclasses import dataclass
from enum import StrEnum

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional

EPSILON = 1e-8
MIN_PROTOTYPES = 2


class Centring(StrEnum):
    EMA = "ema"
    SINKHORN = "sinkhorn"


@dataclass(frozen=True)
class Prototypes:
    k: int = 128
    """Prototypes per head. iBOT spends 8192 on ImageNet, whose 1.28 M images carry ~250 M patch tokens: ~30 k tokens
    per prototype. Our ~1-2 M window steps at that ratio argue for ~50; 128 keeps each prototype ~10 k steps, the same
    order, while still spanning more than the ~40 phonemes and their contexts a 50 ms patch can carry."""
    bottleneck: int = 128
    """Width the head's MLP projects to before the prototypes (DINO: 256)."""
    hidden: int = 256
    """The head MLP's hidden width (DINO: 2048)."""
    student_temp: float = 0.1
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    """The teacher's softmax temperature warms from the start to the end value (DINO: 0.04 to 0.07)."""
    temp_warmup: float = 0.3
    """Share of the steps over which it warms."""
    centring: Centring = Centring.EMA
    """How teacher outputs are kept from collapsing to one prototype: an EMA centre subtracted from the logits (DINO),
    or Sinkhorn-Knopp equalising each prototype's mass over the batch (SwAV, DINOv2)."""
    centre_momentum: float = 0.9
    sinkhorn_iters: int = 3
    perplexity_floor: float = 2.0
    """The teacher's prototype-usage perplexity under which the run aborts (1 is every step on one prototype)."""

    def __post_init__(self) -> None:
        if self.k < MIN_PROTOTYPES:
            raise ValueError(f"{self.k} prototypes cannot be told apart")
        if not 0 <= self.temp_warmup <= 1:
            raise ValueError(f"temp_warmup {self.temp_warmup} is not a share")
        if not 0 <= self.centre_momentum < 1:
            raise ValueError(f"centre_momentum {self.centre_momentum} is not in [0, 1)")

    def teacher_temp(self, progress: float) -> float:
        """The teacher's temperature `progress` (0 to 1) of the way through training."""
        ramp = min(1.0, progress / self.temp_warmup) if self.temp_warmup else 1.0
        return self.teacher_temp_start + (self.teacher_temp_end - self.teacher_temp_start) * ramp


@dataclass(frozen=True)
class Scores:
    """Each row's logits over the prototypes, read at a softmax temperature."""

    logits: Float[Tensor, "n k"]
    temp: float

    def sinkhorn(self, iters: int) -> Float[Tensor, "n k"]:
        """Each row's distribution over prototypes with the batch's mass spread evenly over them: alternately
        normalise prototypes to 1/k and rows to 1/n for `iters` rounds (Cuturi 2013), then rescale rows to sum to 1."""
        n, k = self.logits.shape
        plan = ((self.logits - self.logits.amax()) / self.temp).exp()
        plan = plan / plan.sum()
        for _ in range(iters):
            plan = plan / plan.sum(dim=0, keepdim=True) / k
            plan = plan / plan.sum(dim=1, keepdim=True) / n
        return plan * n

    def cross_entropy(self, probs: Float[Tensor, "n k"]) -> Float[Tensor, ""]:
        """The mean over rows of the cross-entropy of the student's tempered softmax from the teacher's `probs`."""
        return -(probs * functional.log_softmax(self.logits / self.temp, dim=-1)).sum(dim=-1).mean()


@dataclass(frozen=True)
class Assignment:
    """Each row's distribution over the prototypes, and how collapsed they read."""

    probs: Float[Tensor, "n k"]

    def entropy(self) -> Float[Tensor, ""]:
        """The mean row entropy: 0 when each row is one-hot, log k when uniform."""
        return -(self.probs * (self.probs + EPSILON).log()).sum(dim=-1).mean()

    def perplexity(self) -> Float[Tensor, ""]:
        """The effective number of prototypes the rows use: exp of the entropy of their mean, 1 when every row picks
        one prototype, k when they spread evenly."""
        usage = self.probs.mean(dim=0)
        return torch.exp(-(usage * (usage + EPSILON).log()).sum())

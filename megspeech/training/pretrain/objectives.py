"""The terms a pretraining loss is a weighted sum of, each a strategy over what one step read: data2vec's regression
onto the teacher's feature map (Baevski et al. 2022), iBOT's masked-patch distillation and DINO's cross-view
distillation over prototypes (Zhou et al. 2022; Caron et al. 2021), and DINOv2's KoLeo spread regulariser
(Sablayrolles et al. 2019; Oquab et al. 2023)."""

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from operator import add
from typing import ClassVar, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional

from megspeech.training.loop.ema import Ema
from megspeech.training.pretrain.head import Head
from megspeech.training.pretrain.prototypes import Assignment, Centring, Prototypes, Scores

Readings = dict[str, Tensor]


class Loss(StrEnum):
    SMOOTH_L1 = "smooth_l1"
    COSINE = "cosine"


LOSSES: dict[Loss, Callable[[Tensor, Tensor], Tensor]] = {
    Loss.SMOOTH_L1: functional.smooth_l1_loss,
    Loss.COSINE: lambda pred, target: (1 - functional.cosine_similarity(pred, target, dim=-1)).mean(),
}


class Kind(StrEnum):
    REGRESSION = "regression"
    IBOT = "ibot"
    DINO = "dino"
    KOLEO = "koleo"


@dataclass(frozen=True)
class Batch:
    """What one step read, for every term to score."""

    student: list[Float[Tensor, "b h t"]]
    """The student's last feature map of each masked view."""
    hidden: list[Bool[Tensor, "b t"]]
    """Each view's masked time steps."""
    clean: Float[Tensor, "b h t"]
    """The teacher's last feature map of the unmasked window."""
    target: Float[Tensor, "b t h"]
    """The teacher's regression target of the unmasked window."""
    across: list[Float[Tensor, "b h t"]]
    """The teacher's last feature map of each masked view, read only when a term asks."""
    progress: float
    """Share of training done."""


@dataclass(frozen=True)
class Setup:
    """What a term is built from."""

    predictor: nn.Module
    width: int
    prototypes: Prototypes
    loss: Loss
    floor: float
    device: torch.device


class Term(ABC):
    """One weighted term of the loss."""

    views: ClassVar[int] = 1
    """Masked views of each window the term reads."""
    cross: ClassVar[bool] = False
    """Whether it reads the teacher's feature map of each masked view."""

    def __init__(self, name: str, weight: float) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def loss(self, batch: Batch) -> tuple[Tensor, Readings]:
        """The term's unweighted loss over `batch` and the diagnostics it read on the way."""

    def parameters(self) -> list[nn.Parameter]:
        """What the optimiser trains for this term, beyond the encoder."""
        return []

    def follow(self, decay: float) -> None:
        """Move the term's teacher state `1 - decay` of the way to its student's; a term without one keeps it."""
        return None

    def floors(self) -> dict[str, float]:
        """Diagnostics, by name, under which the run is collapsed."""
        return {}


class Regression(Term):
    """The student's readout at masked steps regressed onto the teacher's layer-normalised feature map."""

    def __init__(self, name: str, weight: float, setup: Setup) -> None:
        super().__init__(name, weight)
        self.predictor = setup.predictor
        self.measure = LOSSES[setup.loss]
        self.floor = setup.floor

    @staticmethod
    def spread(rows: Float[Tensor, "n h"]) -> Float[Tensor, " 2"]:
        """How varied unit-normalised `rows` are: their per-dimension std over `n` against an isotropic cloud's
        `h ** -0.5` (1 there, 0 when every row is the same), and the norm of their mean."""
        unit = functional.normalize(rows.float(), dim=-1)
        return torch.stack([unit.std(dim=0).mean() * math.sqrt(unit.shape[-1]), unit.mean(dim=0).norm()])

    @override
    def loss(self, batch: Batch) -> tuple[Tensor, Readings]:
        hidden = batch.hidden[0]
        pred = self.predictor(batch.student[0])
        loss = self.measure(pred[hidden].float(), batch.target[hidden])
        spread, (target_spread, centre) = self.spread(pred[hidden].detach()), self.spread(batch.target[hidden])
        return loss, {"spread": spread[0], "target_spread": target_spread, "target_centre": centre}

    @override
    def floors(self) -> dict[str, float]:
        return {"spread": self.floor, "target_spread": self.floor}


class Distillation(Term):
    """A student head's distribution over prototypes cross-entropied from an EMA teacher head's, the teacher's
    outputs kept from collapsing by `Prototypes.centring`."""

    def __init__(self, name: str, weight: float, setup: Setup) -> None:
        super().__init__(name, weight)
        self.config = setup.prototypes
        self.head = Head(setup.width, self.config).to(setup.device)
        self.teacher = copy.deepcopy(self.head).requires_grad_(requires_grad=False)
        self.centre = torch.zeros(self.config.k, device=setup.device)
        self.assign: dict[Centring, Callable[[Tensor, float], Tensor]] = {
            Centring.EMA: self.centred,
            Centring.SINKHORN: self.balanced,
        }

    @override
    def parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    @override
    @torch.no_grad()
    def follow(self, decay: float) -> None:
        Ema(decay, Ema.state(self.teacher)).step(self.head, 0.0)

    @override
    def floors(self) -> dict[str, float]:
        return {f"{self.name}/perplexity": self.floor}

    @property
    def floor(self) -> float:
        return self.config.perplexity_floor

    def centred(self, logits: Float[Tensor, "n k"], temp: float) -> Float[Tensor, "n k"]:
        """The softmax of `logits` less the EMA centre."""
        return ((logits - self.centre) / temp).softmax(dim=-1)

    def balanced(self, logits: Float[Tensor, "n k"], temp: float) -> Float[Tensor, "n k"]:
        """`logits` as Sinkhorn-Knopp balances them over prototypes."""
        return Scores(logits, temp).sinkhorn(self.config.sinkhorn_iters)

    @torch.no_grad()
    def teach(
        self, features: Float[Tensor, "n w"], progress: float, keep: Bool[Tensor, " n"] | None = None
    ) -> Float[Tensor, "m k"]:
        """The teacher's distribution over prototypes for `features`' `keep` rows (all by default); the EMA centre
        then moves toward the mean logits of every row."""
        logits = self.teacher(features)
        chosen = logits if keep is None else logits[keep]
        probs = self.assign[self.config.centring](chosen, self.config.teacher_temp(progress))
        momentum = self.config.centre_momentum
        self.centre = momentum * self.centre + (1 - momentum) * logits.mean(dim=0)
        return probs

    def readings(self, probs: Float[Tensor, "n k"]) -> Readings:
        """The teacher's mean entropy and its prototype-usage perplexity over `probs`."""
        assignment = Assignment(probs)
        return {f"{self.name}/entropy": assignment.entropy(), f"{self.name}/perplexity": assignment.perplexity()}

    def distil(
        self, student: Float[Tensor, "n w"], teacher: Float[Tensor, "n k"], keep: Bool[Tensor, " n"] | None = None
    ) -> Float[Tensor, ""]:
        """The cross-entropy of the student's `keep` rows from the teacher's distribution."""
        logits = self.head(student)
        return Scores(logits if keep is None else logits[keep], self.config.student_temp).cross_entropy(teacher)


class IBot(Distillation):
    """iBOT's masked-patch objective: at each masked step the student, which cannot see it, matches the teacher's
    distribution over prototypes there in the unmasked window."""

    @override
    def loss(self, batch: Batch) -> tuple[Tensor, Readings]:
        keep = batch.hidden[0].reshape(-1)
        probs = self.teach(batch.clean.transpose(1, 2).reshape(-1, batch.clean.shape[1]), batch.progress, keep)
        student = batch.student[0].transpose(1, 2).reshape(-1, batch.clean.shape[1])
        return self.distil(student, probs, keep), self.readings(probs)


class Dino(Distillation):
    """DINO's global objective over each window's mean feature: view A's student distribution matches the teacher's
    on view B and the reverse."""

    views: ClassVar[int] = 2
    cross: ClassVar[bool] = True

    @override
    def loss(self, batch: Batch) -> tuple[Tensor, Readings]:
        pooled = [maps.mean(dim=-1) for maps in batch.across]
        probs = self.teach(torch.cat(pooled), batch.progress)
        first, second = probs.chunk(2)
        students = [maps.mean(dim=-1) for maps in batch.student]
        loss = 0.5 * (self.distil(students[0], second) + self.distil(students[1], first))
        return loss, self.readings(probs)


class KoLeo(Term):
    """DINOv2's Kozachenko-Leonenko regulariser: spreads the unit-normalised mean feature of each window in the batch
    by maximising the log distance to its nearest neighbour."""

    @staticmethod
    def spread(features: Float[Tensor, "n w"]) -> Float[Tensor, " n"]:
        """Each row's distance to the nearest other row once unit-normalised."""
        unit = functional.normalize(features.float(), dim=-1)
        self_pairs = torch.eye(len(unit), dtype=torch.bool, device=unit.device)
        return torch.cdist(unit, unit).masked_fill(self_pairs, math.inf).amin(dim=1)

    @override
    def loss(self, batch: Batch) -> tuple[Tensor, Readings]:
        nearest = self.spread(batch.student[0].mean(dim=-1))
        return -(nearest + 1e-8).log().mean(), {"koleo/nearest": nearest.detach().mean()}


BUILDERS: dict[Kind, Callable[[str, float, Setup], Term]] = {
    Kind.REGRESSION: Regression,
    Kind.IBOT: IBot,
    Kind.DINO: Dino,
    Kind.KOLEO: lambda name, weight, _: KoLeo(name, weight),
}


@dataclass(frozen=True)
class Objective:
    """One term of the loss and its weight; `kind:weight` on the command line."""

    kind: Kind
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"{self.kind} weight {self.weight} must be positive")

    @classmethod
    def parse(cls, text: str) -> "Objective":
        """`ibot` or `ibot:0.5`."""
        kind, _, weight = text.partition(":")
        return cls(Kind(kind), float(weight) if weight else 1.0)

    def build(self, setup: Setup) -> Term:
        return BUILDERS[self.kind](self.kind.value, self.weight, setup)


@dataclass(frozen=True)
class Terms:
    """The loss's terms, read and trained as one."""

    terms: tuple[Term, ...]

    @classmethod
    def of(cls, objectives: tuple[Objective, ...], setup: Setup) -> "Terms":
        return cls(tuple(objective.build(setup) for objective in objectives))

    @property
    def views(self) -> int:
        """Masked views of each window the hungriest term reads."""
        return max(term.views for term in self.terms)

    @property
    def cross(self) -> bool:
        """Whether any term reads the teacher's feature map of each masked view."""
        return any(term.cross for term in self.terms)

    def parameters(self) -> list[nn.Parameter]:
        return [parameter for term in self.terms for parameter in term.parameters()]

    def floors(self) -> dict[str, float]:
        return {name: floor for term in self.terms for name, floor in term.floors().items()}

    def follow(self, decay: float) -> None:
        for term in self.terms:
            term.follow(decay)

    def clear(self) -> None:
        """Drop the gradients of every term's parameters."""
        for parameter in self.parameters():
            parameter.grad = None

    def score(self, batch: Batch) -> tuple[Tensor, Readings]:
        """The weighted total over `batch`, and each term's unweighted loss (`loss/<term>`) then every diagnostic."""
        scored = [(term, *term.loss(batch)) for term in self.terms]
        total = reduce(add, (term.weight * loss for term, loss, _ in scored))
        losses = {f"loss/{term.name}": loss for term, loss, _ in scored}
        return total, losses | {key: value for _, _, read in scored for key, value in read.items()}

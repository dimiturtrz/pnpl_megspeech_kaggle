"""A PCA-then-LDA linear reader of feature rows: the PCA width picked on an inner fold of the training runs, the LDA
a Ledoit-Wolf-shrunk pooled covariance under a uniform class prior, scored by top-10 balanced accuracy."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.studies.templates import ABSENT, Noise

LOG = logging.getLogger(__name__)

INNER_STRIDE = 10
DIM_FRACTIONS = (0.05, 0.1, 0.2, 0.34, 0.52, 0.63, 0.79, 0.89, 0.98)
"""PCA widths tried on the inner fold, as shares of the feature width."""


@dataclass(frozen=True)
class Dataset:
    """Feature rows, their word labels and the index of the run each came from."""

    x: Float[Tensor, "n d"]
    y: Int[Tensor, " n"]
    group: Int[Tensor, " n"]

    def take(self, mask: Bool[Tensor, " n"]) -> Self:
        return type(self)(self.x[mask], self.y[mask], self.group[mask])


@dataclass(frozen=True)
class Pca:
    mean: Float[Tensor, " d"]
    axes: Float[Tensor, "d d"]
    """Principal axes as columns, by decreasing variance."""

    @classmethod
    def fit(cls, x: Float[Tensor, "n d"]) -> Self:
        mean = x.mean(dim=0)
        centred = x - mean
        _, vectors = torch.linalg.eigh((centred.T @ centred).double())
        return cls(mean, vectors.flip(1).float())

    def project(self, x: Float[Tensor, "n d"], dims: int) -> Float[Tensor, "n k"]:
        return (x - self.mean) @ self.axes[:, :dims]


@dataclass(frozen=True)
class Lda:
    """Class means under one Ledoit-Wolf-shrunk pooled covariance, a uniform class prior."""

    means: Float[Tensor, "v k"]
    precision: Float[Tensor, "k k"]
    present: Tensor

    @classmethod
    def fit(cls, z: Float[Tensor, "n k"], y: Int[Tensor, " n"]) -> Self:
        counts = torch.bincount(y, minlength=VOCAB_SIZE)
        sums = torch.zeros(VOCAB_SIZE, z.shape[1]).index_add_(0, y, z)
        means = sums / counts.clamp(min=1)[:, None]
        residual = z - means[y]
        total = (residual.T @ residual).double()
        quartic = residual.pow(2).sum(dim=1).double().pow(2).sum()
        cov = Noise.shrunk(total, quartic, torch.tensor(float(len(z)), dtype=torch.float64))
        return cls(means, torch.linalg.inv(cov).float(), counts > 0)

    def scores(self, z: Float[Tensor, "n k"]) -> Float[Tensor, "n v"]:
        """Every word's discriminant per row; `ABSENT` for a word with no training row."""
        weights = self.means @ self.precision
        bias = 0.5 * (weights * self.means).sum(dim=1)
        return (z @ weights.T - bias).masked_fill(~self.present, ABSENT)


@dataclass(frozen=True)
class Outcome:
    """The PCA width the inner fold chose and each domain's novel-span top-10 balanced accuracy."""

    dims: int
    domains: dict[str, float]

    @property
    def macro(self) -> float:
        return float(np.mean(list(self.domains.values())))


@dataclass(frozen=True)
class Harness:
    fractions: tuple[float, ...] = DIM_FRACTIONS

    def dims(self, width: int) -> list[int]:
        return sorted({max(1, round(share * width)) for share in self.fractions})

    @staticmethod
    def balanced(scores: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> float:
        ruler = Ruler()
        return ruler.balanced(ruler.hits(scores, y), y)

    def decoded(self, fit: Dataset, test: Dataset, dims: int, pca: Pca) -> float:
        lda = Lda.fit(pca.project(fit.x, dims), fit.y)
        return self.balanced(lda.scores(pca.project(test.x, dims)), test.y)

    def inner(self, train: Dataset) -> int:
        """The PCA width of best balanced accuracy on every `INNER_STRIDE`th training run, fit on the rest."""
        held = train.group % INNER_STRIDE == 0
        fit, test = train.take(~held), train.take(held)
        pca = Pca.fit(fit.x)
        found = {dims: self.decoded(fit, test, dims, pca) for dims in self.dims(train.x.shape[1])}
        LOG.info("inner fold %s", " ".join(f"{dims}:{score:.4f}" for dims, score in found.items()))
        return max(found, key=lambda dims: found[dims])

    def outcome(self, train: Dataset, domains: Mapping[str, Dataset]) -> Outcome:
        dims = self.inner(train)
        pca = Pca.fit(train.x)
        lda = Lda.fit(pca.project(train.x, dims), train.y)
        return Outcome(
            dims, {name: self.balanced(lda.scores(pca.project(d.x, dims)), d.y) for name, d in domains.items()}
        )

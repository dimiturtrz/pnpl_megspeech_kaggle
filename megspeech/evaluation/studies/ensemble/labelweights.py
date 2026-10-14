"""Whether weighting ensemble members per label beats the uniform blend, cross-fit across the Broad ruler's subjects.

Each member's Broad-ruler log-probabilities (unseen-span words of subjects 1-32) are pooled; the subjects are split in
two halves, a per-label weight `w[m, l] = softmax_m(beta * shrunk[m, l])` is fit on one half and scores the other, and
the halves swap. `shrunk[m, l] = (n_l * s[m, l] + k * s[m]) / (n_l + k)` shrinks a member's mean true-label
log-probability on label `l` toward its overall one. `(k, beta)` is chosen on the fit half alone, by a second cross-fit
inside it. Both blends are scored by the ruler's own metric (macro balanced top-10 at the macro's best adjustment).

    uv run python -m megspeech.evaluation.studies ensemble.labelweights NAME [NAME ...] --device cpu --aligned
"""

import argparse
import logging
from dataclasses import dataclass, replace
from itertools import product

import torch

from core.paths import DataRoot
from megspeech.device import Device
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.labelweighted import Pool, Shrinkage
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

SHRINKS = (5.0, 20.0, 80.0)
"""Words of pseudo-evidence the overall score lends each label; the median label has ~13 words per domain."""
BETAS = (0.0, 10.0, 30.0, 100.0)
"""Softmax sharpness over members; 0 is the uniform blend."""


@dataclass(frozen=True)
class Fold:
    """One fit/held split: what the fit half chose and how both blends score on the held half."""

    chosen: Shrinkage
    uniform: float
    weighted: float
    every: dict[Shrinkage, float]


@dataclass(frozen=True)
class CrossFit:
    pool: Pool

    @staticmethod
    def halves(subjects: list[int]) -> tuple[list[int], list[int]]:
        return subjects[::2], subjects[1::2]

    @staticmethod
    def grid() -> list[Shrinkage]:
        return [Shrinkage(k, beta) for k, beta in product(SHRINKS, BETAS)]

    def held_out(self, fit: list[int], held: list[int], shrinkage: Shrinkage) -> float:
        """Macro on `held` of weights fit on `fit`."""
        return self.pool.macro(held, shrinkage.weights(self.pool, fit)).bacc10

    def inner(self, fit: list[int], shrinkage: Shrinkage) -> float:
        """`shrinkage`'s mean held-out macro across the two halves of `fit`."""
        a, b = self.halves(fit)
        return (self.held_out(a, b, shrinkage) + self.held_out(b, a, shrinkage)) / 2

    def choose(self, fit: list[int]) -> Shrinkage:
        """The grid point scoring best under a cross-fit inside `fit`: the held half is never seen."""
        return max(self.grid(), key=lambda shrinkage: self.inner(fit, shrinkage))

    def fold(self, fit: list[int], held: list[int]) -> Fold:
        chosen = self.choose(fit)
        every = {shrinkage: self.held_out(fit, held, shrinkage) for shrinkage in self.grid()}
        uniform = self.pool.macro(held, self.pool.uniform()).bacc10
        return Fold(chosen, uniform, every[chosen], every)

    def run(self) -> list[Fold]:
        a, b = self.halves(self.pool.subjects())
        return [self.fold(a, b), self.fold(b, a)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, each one member")
    parser.add_argument("--device", default=Device.CPU)
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    scoring = replace(Scoring(fallback=Fallback(Unseen.DEEP)), aligned=args.aligned)
    pool = Pool.read(Group(DataRoot.from_config(), scoring, torch.device(args.device)), args.names)
    full = pool.macro(pool.subjects(), pool.uniform())
    LOG.info("uniform blend, full ruler %.4f tau %g over %d words", full.bacc10, full.tau, full.n)
    folds = CrossFit(pool).run()
    for index, fold in enumerate(folds):
        delta = fold.weighted - fold.uniform
        LOG.info(
            "fold %d  chosen k %g beta %g  uniform %.4f  weighted %.4f  delta %+.4f",
            index,
            fold.chosen.k,
            fold.chosen.beta,
            fold.uniform,
            fold.weighted,
            delta,
        )
        for shrinkage, score in fold.every.items():
            LOG.info(
                "  fold %d  k %g beta %g  held %.4f (%+.4f)",
                index,
                shrinkage.k,
                shrinkage.beta,
                score,
                score - fold.uniform,
            )
    uniform = sum(f.uniform for f in folds) / len(folds)
    weighted = sum(f.weighted for f in folds) / len(folds)
    LOG.info("2-fold mean  uniform %.4f  weighted %.4f  delta %+.4f", uniform, weighted, weighted - uniform)


if __name__ == "__main__":
    main()

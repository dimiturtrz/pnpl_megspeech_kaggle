"""Whether weighting ensemble members per label beats the uniform blend, cross-fit across the Broad ruler's subjects.

Each member's Broad-ruler log-probabilities (unseen-span words of subjects 1-32) are pooled; the subjects are split in
two halves, a per-label weight `w[m, l] = softmax_m(beta * shrunk[m, l])` is fit on one half and scores the other, and
the halves swap. `shrunk[m, l] = (n_l * s[m, l] + k * s[m]) / (n_l + k)` shrinks a member's mean true-label
log-probability on label `l` toward its overall one. `(k, beta)` is chosen on the fit half alone, by a second cross-fit
inside it. Both blends are scored by the ruler's own metric (macro balanced top-10 at the macro's best adjustment).

    uv run python -m megspeech.evaluation.studies.labelweights NAME [NAME ...] --device cpu --aligned
"""

import argparse
import logging
from dataclasses import dataclass, replace
from itertools import product

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.rulers.broad import BroadRuler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Score
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

SHRINKS = (5.0, 20.0, 80.0)
"""Words of pseudo-evidence the overall score lends each label; the median label has ~13 words per domain."""
BETAS = (0.0, 10.0, 30.0, 100.0)
"""Softmax sharpness over members; 0 is the uniform blend."""


@dataclass(frozen=True)
class Shrinkage:
    """A member weighting: `beta` sharpens, `k` pulls each label's score toward the member's overall one."""

    k: float
    beta: float

    def weights(self, pool: "Pool", subjects: list[int]) -> Float[Tensor, "m v"]:
        """Per-label member weights fit on `subjects`' words."""
        y = torch.cat([pool.labels[s] for s in subjects])
        logp = torch.cat([pool.logp[s] for s in subjects], dim=1)
        true = logp.gather(2, y[None, :, None].expand(len(logp), -1, 1)).squeeze(2)
        count = torch.bincount(y, minlength=logp.shape[2]).float()
        total = torch.zeros(len(logp), logp.shape[2]).index_add_(1, y, true)
        shrunk = (total + self.k * true.mean(dim=1, keepdim=True)) / (count + self.k)
        return torch.softmax(self.beta * shrunk, dim=0)


@dataclass(frozen=True)
class Pool:
    """Every member's log-probabilities for each subject's unseen-span words, with their labels."""

    labels: dict[int, Int[Tensor, " n"]]
    logp: dict[int, Float[Tensor, "m n v"]]
    prior: Float[Tensor, " v"]
    evaluator: Evaluator

    @classmethod
    def read(  # devtools-ignore: test-mirror
        cls, root: DataRoot, names: list[str], scoring: Scoring, device: torch.device
    ) -> "Pool":
        """Each name loaded alone and read on the Broad ruler's runs; every member must read the same words."""
        runs = Run.discover(root.raw())
        labels: dict[int, Int[Tensor, " n"]] = {}
        reads: dict[int, list[Float[Tensor, "n v"]]] = {}
        priors: list[Float[Tensor, " v"]] = []
        evaluator = None
        for name in names:
            loaded = Loader.of(root, [name], scoring, device)
            evaluator = evaluator or loaded.evaluator
            priors.append(loaded.log_prior)
            for run in BroadRuler.scored(loaded, runs):
                split, context = loaded.run_context(run)
                novel = split.novel(loaded.evaluator.seen)
                if run.subject in labels and not torch.equal(labels[run.subject], split.y[novel]):
                    raise ValueError(f"{name} reads different words for subject {run.subject}")
                labels[run.subject] = split.y[novel]
                reads.setdefault(run.subject, []).append(context[novel])
            LOG.info("read %s", name)
        if evaluator is None:
            raise ValueError("no members")
        return cls(labels, {s: torch.stack(r) for s, r in reads.items()}, torch.stack(priors).mean(0), evaluator)

    def subjects(self) -> list[int]:
        return sorted(self.labels)

    def blended(self, subject: int, weights: Float[Tensor, "m v"]) -> Float[Tensor, "n v"]:
        """The members' log-probabilities for `subject`, each label's column weighted by `weights`."""
        return (weights[:, None, :] * self.logp[subject]).sum(0)

    def uniform(self) -> Float[Tensor, "m v"]:
        members, _, vocabulary = self.logp[self.subjects()[0]].shape
        return torch.full((members, vocabulary), 1.0 / members)

    def macro(self, subjects: list[int], weights: Float[Tensor, "m v"]) -> Score:
        """The ruler's macro over `subjects` of the blend under `weights`."""
        scores = {s: self.evaluator.evaluate(self.blended(s, weights), self.labels[s], self.prior) for s in subjects}
        return Score.mean(self.evaluator.counted(scores).values())


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    scoring = replace(Scoring(fallback=Fallback(Unseen.DEEP)), aligned=args.aligned)
    pool = Pool.read(DataRoot.from_config(), args.names, scoring, torch.device(args.device))
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

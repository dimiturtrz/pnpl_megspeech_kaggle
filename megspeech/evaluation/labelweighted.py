"""Ensemble members weighted per label: each member's Broad-ruler log-probabilities pooled, and the shrunk softmax
weights `w[m, l] = softmax_m(beta * shrunk[m, l])` fit on any subjects of them.

`shrunk[m, l] = (n_l * s[m, l] + k * s[m]) / (n_l + k)` shrinks a member's mean true-label log-probability on label `l`
toward its overall one.
"""

import logging
import math
from dataclasses import dataclass, field, replace

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.cache.kaggle import Track
from megspeech.data.raw.runs import Run
from megspeech.evaluation.group import Group
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Score

LOG = logging.getLogger(__name__)


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

    @staticmethod
    def entropy(weights: Float[Tensor, "m v"]) -> Float[Tensor, " v"]:
        """Each label's weight entropy as a fraction of the uniform blend's: 1 is uniform, 0 is one member alone."""
        members = len(weights)
        return -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(0) / math.log(members)


@dataclass
class Reads:
    """Members' Broad-ruler reads gathered one member at a time: each subject's labels, which every member must share,
    and every member's log-probabilities for them."""

    labels: dict[int, Int[Tensor, " n"]] = field(default_factory=dict)
    logp: dict[int, list[Float[Tensor, "n v"]]] = field(default_factory=dict)

    def add(self, name: str, subject: int, labels: Int[Tensor, " n"], logp: Float[Tensor, "n v"]) -> None:
        """Member `name`'s `logp` for `subject`'s words `labels`; a member reading other words is a `ValueError`."""
        known = self.labels.setdefault(subject, labels)
        if not torch.equal(known, labels):
            raise ValueError(f"{name} reads different words for subject {subject}")
        self.logp.setdefault(subject, []).append(logp)

    def stacked(self) -> dict[int, Float[Tensor, "m n v"]]:
        """Each subject's members' log-probabilities, member first."""
        return {subject: torch.stack(members) for subject, members in self.logp.items()}


@dataclass(frozen=True)
class Pool:
    """Every member's log-probabilities for each subject's unseen-span words, with their labels; `kaggle` holds each
    member's log-probabilities for the Kaggle test when a track was asked for."""

    labels: dict[int, Int[Tensor, " n"]]
    logp: dict[int, Float[Tensor, "m n v"]]
    prior: Float[Tensor, " v"]
    evaluator: Evaluator
    kaggle: Float[Tensor, "m r v"] | None = None
    kaggle_subject: Int[Tensor, " r"] | None = None
    """The subject each Kaggle test row is read from, for a per-subject balance."""

    @classmethod
    def read(  # devtools-ignore: test-mirror
        cls, group: Group, names: list[str], track: Track | None = None
    ) -> "Pool":
        """Each name loaded alone as `group` loads it and read on the Broad ruler's runs; every member must read the
        same words."""
        runs = Run.discover(group.root.raw())
        reads = Reads()
        test_subject: Int[Tensor, " r"] | None = None
        priors: list[Float[Tensor, " v"]] = []
        tests: list[Float[Tensor, "r v"]] = []
        seen: frozenset[tuple[int, ...]] = frozenset()
        evaluator = None
        for name in names:
            loaded = Loader.of(group, [name])
            evaluator = evaluator or loaded.evaluator
            seen |= loaded.evaluator.seen
            priors.append(loaded.log_prior)
            for run in loaded.broad(runs):
                split, context = loaded.run_context(run)
                novel = split.novel(loaded.evaluator.seen)
                reads.add(name, run.subject, split.y[novel], context[novel])
            if track is not None:
                split, chosen = loaded.kaggle_chosen(track)
                test_subject = split.subject
                tests.append(chosen)
            LOG.info("read %s", name)
        if evaluator is None:
            raise ValueError("no members")
        return cls(
            reads.labels,
            reads.stacked(),
            torch.stack(priors).mean(0),
            replace(evaluator, seen=seen),
            torch.stack(tests) if tests else None,
            test_subject,
        )

    def subjects(self) -> list[int]:
        return sorted(self.labels)

    def blended(self, subject: int, weights: Float[Tensor, "m v"]) -> Float[Tensor, "n v"]:
        """The members' log-probabilities for `subject`, each label's column weighted by `weights`."""
        return (weights[:, None, :] * self.logp[subject]).sum(0)

    def tested(self, weights: Float[Tensor, "m v"]) -> Float[Tensor, "r v"]:
        """The members' Kaggle test log-probabilities, each label's column weighted by `weights`."""
        if self.kaggle is None:
            raise ValueError("the pool was read without a track")
        return (weights[:, None, :] * self.kaggle).sum(0)

    def finished(self, weights: Float[Tensor, "m v"], tau: float) -> Float[Tensor, "r v"]:
        """The Kaggle test's probabilities under `weights`, logit-adjusted by `tau`."""
        return torch.softmax(
            self.evaluator.calibration(self.prior, tau).adjusted(self.tested(weights), self.kaggle_subject), dim=-1
        )

    def uniform(self) -> Float[Tensor, "m v"]:
        members, _, vocabulary = self.logp[self.subjects()[0]].shape
        return torch.full((members, vocabulary), 1.0 / members)

    def macro(self, subjects: list[int], weights: Float[Tensor, "m v"]) -> Score:
        """The ruler's macro over `subjects` of the blend under `weights`."""
        scores = {s: self.evaluator.evaluate(self.blended(s, weights), self.labels[s], self.prior) for s in subjects}
        return Score.mean(self.evaluator.counted(scores).values())

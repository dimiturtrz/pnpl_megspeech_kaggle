"""The local Broad ruler: each subject 1-32 scored on its own Sherlock1 session-11 run (`Partition.POOL`).

That run's text is subject 0's validation run's, which no model trains on, while subjects 1-32 train on session 12,
so it is novel text in every subject's own MEG. A subject's score is balanced top-10 over its unseen sentence spans,
as the Deep ruler scores a domain; the ruler is the unweighted mean over subjects at one adjustment for all.

`--reading unseen` reads every subject as the holdout's zero-shot subjects 33-39 are read — through the `--unseen`
fallback instead of its own subject matrix — so the fallbacks can be compared where the answers are known.

    uv run python -m megspeech.evaluation.broad sent40-s7 [--reading unseen --unseen mean]
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import torch
from jaxtyping import Int
from pnpl.competition.holdout import N_SUBJECTS
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.corpus import Corpus, DataConfig
from megspeech.data.grouping import Grouping
from megspeech.data.runs import Partition, Run
from megspeech.data.split import Split
from megspeech.evaluation.pooling import Pooling
from megspeech.evaluation.report import Score
from megspeech.model.subjects import Unseen

LOG = logging.getLogger(__name__)


class Reading(StrEnum):
    """Which subject a run's words are presented to the model as."""

    OWN = "own"
    UNSEEN = "unseen"
    """A subject past every trained one, as a zero-shot holdout subject is."""

    def subjects(self, subject: Int[Tensor, " n"]) -> Int[Tensor, " n"]:
        return READINGS[self](subject)


READINGS: dict[Reading, Callable[[Tensor], Tensor]] = {
    Reading.OWN: lambda subject: subject,
    Reading.UNSEEN: lambda subject: torch.full_like(subject, N_SUBJECTS),
}


@dataclass(frozen=True)
class BroadRuler:
    """A trained model scored subject by subject on each one's pooled run, every subject read as `reading` says."""

    pooling: Pooling
    reading: Reading = Reading.OWN

    def split(self, run: Run) -> Split:
        """`run`'s words cut as the model's grouping cuts them, each presented as `reading`'s subject."""
        source, words = self.pooling.load([run])
        split = Corpus.split(source, words, self.pooling.data, np.ones(len(words)))
        return replace(split, subject=self.reading.subjects(split.subject))

    def score(self, run: Run) -> Score:
        """`run`'s unseen-span words scored from its own MEG."""
        split = self.split(run)
        context = self.pooling.context(split)
        return self.pooling.matched(split, context, torch.ones(len(split.y), dtype=torch.bool, device=split.y.device))

    def scores(self, runs: list[Run]) -> dict[int, Score]:
        """Every pooled run's score, by subject in ascending order."""
        pooled = sorted((r for r in runs if r.partition is Partition.POOL), key=lambda r: r.subject)
        return {run.subject: self.score(run) for run in pooled}

    @staticmethod
    def report(scores: dict[int, Score]) -> Score:
        """Log each subject's score at the macro's adjustment, then the macro, and return it."""
        macro = Score.mean(scores.values())
        for subject, score in scores.items():
            LOG.info("subject %2d  bacc10 %.4f  n %d", subject, score.by_tau[macro.tau], score.n)
        LOG.info(
            "broad ruler %.4f tau %g over %d subjects, %d words  by tau %s",
            macro.bacc10,
            macro.tau,
            len(scores),
            macro.n,
            {tau: round(value, 4) for tau, value in macro.by_tau.items()},
        )
        return macro


@dataclass(frozen=True)
class Options:
    """What the command line chose: the model, how it reads unseen subjects, and how every subject is read."""

    name: str
    grouping: Grouping
    unseen: Unseen
    reading: Reading

    @classmethod
    def parse(cls, description: str | None) -> "Options":
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("name", help="a trained model under models/")
        parser.add_argument("--grouping", type=Grouping, choices=list(Grouping), default=DataConfig().grouping)
        parser.add_argument("--unseen", type=Unseen, choices=list(Unseen), default=Unseen.DEEP)
        parser.add_argument("--reading", type=Reading, choices=list(Reading), default=Reading.OWN)
        args = parser.parse_args()
        return cls(args.name, args.grouping, args.unseen, args.reading)

    def ruler(self, root: DataRoot, device: torch.device) -> BroadRuler:
        return BroadRuler(Pooling.of(root, self.name, self.grouping, device, self.unseen), self.reading)


def main() -> None:
    """Log the Broad ruler for one trained model."""
    options = Options.parse(__doc__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    ruler = options.ruler(root, torch.device("cuda"))
    BroadRuler.report(ruler.scores(Run.discover(root.raw())))


if __name__ == "__main__":
    main()

"""The local Broad ruler: each subject 1-32 scored on its own Sherlock1 session-11 run (`Partition.POOL`).

That run's text is subject 0's validation run's, which no model trains on, while subjects 1-32 train on session 12,
so it is novel text in every subject's own MEG. A subject's score is balanced top-10 over its unseen sentence spans,
as the Deep ruler scores a domain; the ruler is the unweighted mean over subjects at one adjustment for all, over the
subjects with the unseen-span words a Deep domain needs to count (`Evaluator.min_novel`).

`--reading unseen` reads every subject as the Kaggle test's zero-shot subjects 33-39 are read — through the
`--fallback.unseen` matrix instead of its own — so the fallbacks can be compared where the answers are known.
"""

import logging

from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.scoring.score import Score

LOG = logging.getLogger(__name__)


class BroadRuler:
    @staticmethod
    def select(loaded: LoadedModel, runs: list[Run]) -> Score:
        scores = loaded.evaluator.counted({run.subject: loaded.unseen(run) for run in Run.pooled(runs)})
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

    @staticmethod
    def graded(loaded: LoadedModel, runs: list[Run], tau: float) -> Graded:
        """Each counted subject a domain, its run loaded alone as `select` loads it."""
        return Graded(
            {
                name: hits
                for run in Run.pooled(runs)
                for name, hits in loaded.evaluator.graded(
                    loaded.model, {str(run.subject): loaded.splits([run])[run][1]}, loaded.log_prior, tau
                ).domains.items()
            }
        )

"""The local Broad ruler: each subject 1-32 scored on its own Sherlock1 session-11 run (`Partition.POOL`), or on
session 12 for a model that adapts on session 11 (`DataConfig.adapt`).

That run's text is subject 0's validation run's (session 12: its test run's), which no model trains on as another
subject's `adapt` session, so it is novel text in every subject's own MEG. A subject's score is balanced top-10 over
its unseen sentence spans, as the Deep ruler scores a domain; the ruler is the unweighted mean over subjects at one
adjustment for all, over the subjects with the unseen-span words a Deep domain needs to count (`Evaluator.min_novel`).
An adapted model also logs the mean over the subjects outside `data.subjects`, the ones it never trained on.

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
        scores = loaded.evaluator.counted({run.subject: loaded.unseen(run) for run in loaded.broad(runs)})
        macro = Score.mean(scores.values())
        for subject, score in scores.items():
            LOG.info("subject %2d  bacc10 %.4f  n %d", subject, score.by_tau[macro.tau], score.n)
        if loaded.data.adapt:
            outside = [s.by_tau[macro.tau] for subject, s in scores.items() if subject not in loaded.data.subjects]
            LOG.info("held-out subjects %d  bacc10 %.4f", len(outside), Score.average(outside))
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
        parts: list[Graded] = []
        for run in loaded.broad(runs):
            split, context = loaded.run_context(run)
            counted = loaded.evaluator.eligible({str(run.subject): split})
            parts.append(loaded.evaluator.grade(dict.fromkeys(counted, context), counted, loaded.log_prior, tau))
        return Graded({name: hits for part in parts for name, hits in part.domains.items()})

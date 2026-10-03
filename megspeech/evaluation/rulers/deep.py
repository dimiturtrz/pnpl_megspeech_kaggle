"""The local Deep ruler: subject 0's validation domains, the macro over their unseen spans selecting the adjustment."""

import logging

from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.score import Epoch

LOG = logging.getLogger(__name__)


class DeepRuler:
    @staticmethod
    def select(loaded: LoadedModel, runs: list[Run]) -> Epoch:
        report = loaded.report(runs)
        LOG.info(
            "%s", {task: (round(r.context.bacc10, 4), round(r.novel.bacc10, 4)) for task, r in report.domains.items()}
        )
        LOG.info("macro %.4f tau %s over %d words", report.macro.bacc10, report.tau, report.macro.n)
        return report

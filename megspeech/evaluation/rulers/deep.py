"""The local Deep ruler: subject 0's val domains, the macro over their unseen spans selecting the adjustment."""

import logging
from typing import override

from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.rulers.base import TrackRuler
from megspeech.evaluation.scoring.score import Epoch

LOG = logging.getLogger(__name__)


class DeepRuler(TrackRuler):
    @override
    def select(self) -> Epoch:
        report = self.loaded.report(self.runs)
        LOG.info(
            "%s", {task: (round(r.context.bacc10, 4), round(r.novel.bacc10, 4)) for task, r in report.domains.items()}
        )
        LOG.info("val macro %.4f tau %s over %d words", report.macro.bacc10, report.tau, report.macro.n)
        novel = [r.novel for r in report.domains.values()]
        LOG.info(
            "novel-text macro %.4f over %d words", sum(s.bacc10 for s in novel) / len(novel), sum(s.n for s in novel)
        )
        return report

    @override
    def graded(self, tau: float) -> Graded:
        return self.loaded.graded(self.runs, tau)

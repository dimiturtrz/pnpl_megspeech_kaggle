"""A model's local test: the labelled sessions kept out of val scored at the adjustment val selects.

Val picks tau (as `megspeech.submission` does); the local test only reports, once, at that tau. Every counted domain's
balanced accuracy and unseen-span word count, the macro and its bootstrap interval, under each span cut.

    uv run python -m megspeech.submission ens-sent40-s0123 --out x --local [--grouping kaggle]
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.metric.ruler import Graded, Ruler

LOG = logging.getLogger(__name__)

SCORED = {Partition.VALIDATION: "val", Partition.TEST: "local test"}
SECTIONS = {Partition.VALIDATION: "official-val", Partition.TEST: "local-test"}
"""The sections of `probs.pt` each partition's predictions are saved under."""


@dataclass(frozen=True)
class LocalTest:
    """`loaded` scored on val and on the local test of `runs`, at the tau its val macro selects."""

    loaded: LoadedModel
    runs: list[Run]
    ruler: Ruler = Ruler()

    def tau(self) -> float:
        return self.loaded.report(self.runs).tau

    def graded(self, partition: Partition, tau: float) -> Graded:
        return replace(self.loaded, partition=partition).graded(self.runs, tau)

    def summary(self, graded: Graded) -> str:
        """The macro with its interval, then each domain's balanced accuracy and word count."""
        interval = self.ruler.interval(self.ruler.macro(graded))
        domains = {
            name: f"{self.ruler.balanced(hits, labels):.4f} n={len(labels)}"
            for name, (hits, labels) in graded.domains.items()
        }
        words = sum(len(labels) for _, labels in graded.domains.values())
        return f"macro {self.ruler.point(graded):.4f} [{interval.low:.4f}, {interval.high:.4f}] n={words} {domains}"

    def record(self, partition: Partition, tau: float, path: Path) -> Path:
        """Every labelled word of `partition`'s domains with its probabilities at `tau`, saved into `path`."""
        loaded = replace(self.loaded, partition=partition)
        return loaded.probabilities_of(self.runs, tau).save(path, SECTIONS[partition])

    def report(self, label: str, probabilities: Path | None = None) -> None:
        """Log both partitions' scores at val's tau, and save their per-word probabilities when `probabilities`
        names a file."""
        tau = self.tau()
        for partition in (Partition.VALIDATION, Partition.TEST):
            LOG.info("%s %s tau %s: %s", label, SCORED[partition], tau, self.summary(self.graded(partition, tau)))
            if probabilities is not None:
                self.record(partition, tau, probabilities)

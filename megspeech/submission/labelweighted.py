"""A Kaggle submission whose members are weighted per label (`megspeech.evaluation.labelweighted`), the weights fit on
every subject of the Broad ruler, whose labels are all known.

    uv run python -m megspeech.submission NAME [NAME ...] --track broad --labelweights K BETA --out NAME
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from megspeech.data.cache.kaggle import Track
from megspeech.evaluation.group import Group
from megspeech.evaluation.labelweighted import Pool, Shrinkage
from megspeech.submission.report import SUBMISSIONS

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelWeighted:
    """`pool`'s members blended under `shrinkage`'s weights, written as `track`'s Kaggle submission."""

    pool: Pool
    shrinkage: Shrinkage
    track: Track

    @classmethod
    def read(  # devtools-ignore: test-mirror
        cls, group: Group, names: list[str], shrinkage: Shrinkage
    ) -> "LabelWeighted":
        """The Broad track's members, loaded as `group` loads them, read once: the ruler's words and the Kaggle
        test."""
        return cls(Pool.read(group, names, Track.BROAD), shrinkage, Track.BROAD)

    def write(self, raw: Path, out: Path) -> Path:
        """The Kaggle test's rows under the weights fit on the whole ruler, adjusted at the ruler's best `tau` for
        that weighted blend; the submission's path."""
        subjects = self.pool.subjects()
        weights = self.shrinkage.weights(self.pool, subjects)
        score = self.pool.macro(subjects, weights)
        entropy = self.shrinkage.entropy(weights)
        LOG.info(
            "in-sample weighted ruler (weights fit on these same %d subjects, not held out) %.4f tau %g over %d words",
            len(subjects),
            score.bacc10,
            score.tau,
            score.n,
        )
        LOG.info(
            "weight entropy / uniform  mean %.3f min %.3f max %.3f  mean top weight %.3f",
            entropy.mean(),
            entropy.min(),
            entropy.max(),
            weights.max(dim=0).values.mean(),
        )
        out.mkdir(parents=True, exist_ok=True)
        return self.track.write(
            raw, self.pool.finished(weights, score.tau).cpu().numpy(), out / SUBMISSIONS[self.track]
        )

"""The leaderboard metric, computed locally: balanced top-k accuracy over the words present."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

TOP = 10


@dataclass(frozen=True)
class BalancedTopK:
    """Mean over present classes of how often the true word ranks in the top `k` (ties counted as hits)."""

    k: int = TOP

    def __call__(self, scores: Float[np.ndarray, "n v"], labels: Int[np.ndarray, " n"]) -> float:
        true = np.take_along_axis(scores, labels[:, None], axis=1)
        hit = (scores > true).sum(axis=1) < self.k
        classes = np.unique(labels)
        return float(np.mean([hit[labels == label].mean() for label in classes]))

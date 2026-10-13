"""How far submissions move from a reference: the share of rows whose top-10 primary words differ as a set, and the
mean overlap of those sets. Only the primary columns are scored, so only they are compared.

    uv run python -m megspeech.submission.compare reference.csv candidate.csv [candidate.csv ...]
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int

from megspeech.data.labels.vocabulary import PRIMARY

LOG = logging.getLogger(__name__)
TOP = 10


@dataclass(frozen=True)
class TopTen:
    """Each row's ten highest-scored primary words, as a membership mask over the vocabulary."""

    index: Int[np.ndarray, " n"]
    chosen: Bool[np.ndarray, "n v"]

    @classmethod
    def read(cls, path: Path) -> "TopTen":
        frame = pd.read_csv(path).sort_values("index")
        scores = frame[list(PRIMARY.words)].to_numpy()
        chosen = np.zeros(scores.shape, dtype=bool)
        np.put_along_axis(chosen, np.argsort(-scores, axis=1, kind="stable")[:, :TOP], values=True, axis=1)
        return cls(frame["index"].to_numpy(), chosen)

    def moved(self, other: "TopTen") -> float:
        """The share of rows whose top-10 set is not the same set in `other`."""
        return float(np.mean(self.overlap(other) < TOP))

    def overlap(self, other: "TopTen") -> Int[np.ndarray, " n"]:
        if not np.array_equal(self.index, other.index):
            raise ValueError("the submissions score different rows")
        return (self.chosen & other.chosen).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reference = TopTen.read(args.reference)
    for path in args.candidates:
        candidate = TopTen.read(path)
        LOG.info(
            "%s: %.3f rows moved, mean overlap %.2f/%d",
            path.stem,
            candidate.moved(reference),
            candidate.overlap(reference).mean(),
            TOP,
        )


if __name__ == "__main__":
    main()

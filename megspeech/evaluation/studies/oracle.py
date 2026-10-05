"""A plumbing oracle for the experts (megspeech-6ow.3): a teacher directory whose target for each vocabulary word is the
word's mean teacher vector, so it is effectively the word's label. An expert trained on it must read the 50 words off
MEG about as well as the base does, or the expert pipeline, not the teacher, is what fails.

    uv run python -m megspeech.evaluation.studies.oracle gpt2-large
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.paths import DataRoot
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.labels.vocabulary import ABSENT, IGNORE, PRIMARY
from megspeech.evaluation.studies.expertcheck import Centroids
from megspeech.objective.targets.speech import Targets

LOG = logging.getLogger(__name__)


class Oracle:
    """The teacher `source` with every word's target replaced by its vocabulary word's centroid."""

    def __init__(self, source: Path, cache: RunCache) -> None:
        self.source = source
        self.cache = cache

    def words(self) -> pd.DataFrame:
        """Every word of the runs `source` has targets for: `run`, `position` in its run's table, `target` label."""
        table = self.cache.table(sorted(path.stem for path in self.source.glob("*.npy")))
        labels = PRIMARY.labels(table["word"])
        return table.assign(position=table.groupby("run").cumcount(), target=np.where(labels == ABSENT, IGNORE, labels))

    def write(self, destination: Path) -> Centroids:
        """Write each run's targets under `destination`, NaN where a word is outside the vocabulary."""
        words = self.words()
        centroids = Centroids.fit(words, self.source)
        destination.mkdir(parents=True, exist_ok=True)
        for run, group in words.groupby("run"):
            rows = len(np.load(self.source / f"{run}.npy", mmap_mode="r"))
            vectors = np.full((rows, centroids.means.shape[1]), np.nan, dtype=np.float16)
            known = group[group["target"] != IGNORE]
            vectors[known["position"].to_numpy()] = centroids.means[known["target"].to_numpy()].numpy()
            Targets(vectors).save(destination / f"{run}.npy")
        return centroids

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("source", help="the teacher's directory under processed/speech")
        arguments = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        source = Targets.directory(root, arguments.source)
        destination = Targets.directory(root, f"oracle-{arguments.source}")
        centroids = Oracle(source, RunCache.under(root.processed())).write(destination)
        LOG.info("wrote %s: mean norm / std %.2f", destination, centroids.offset)


if __name__ == "__main__":
    Oracle.main()

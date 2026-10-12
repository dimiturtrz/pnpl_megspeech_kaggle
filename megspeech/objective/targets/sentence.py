"""Sentence-level teacher targets (megspeech-6ow.1): each word's target is its sentence's, the mean of the sentence's
word-level targets (`megspeech.objective.targets.speech.Targets`) over the words that have one, so a sentence
expert is read once per sentence rather than once per word.

uv run python -m megspeech.objective.targets.sentence gpt2-large
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Int

from core.paths import DataRoot
from megspeech.data.cache.run_cache import NO_SENTENCE, RunCache
from megspeech.data.config import DataConfig, Val
from megspeech.data.raw.runs import Column, Run
from megspeech.objective.targets.speech import Targets

SUFFIX = "-sent"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sentences:
    """The sentence each word of one run was heard in, `NO_SENTENCE` for none."""

    sentence: Int[np.ndarray, " n"]

    def pool(self, targets: Targets) -> Targets:
        """Every word of a sentence the mean target of the sentence's words that have one, standardised over the run;
        a word with no sentence, or whose sentence has no target, stays absent."""
        vectors = targets.vectors.astype(np.float32)
        frame = pd.DataFrame(np.where(targets.present[:, None], vectors, np.nan))
        mean = frame.groupby(self.sentence).transform("mean").to_numpy()
        pooled = np.where((self.sentence != NO_SENTENCE)[:, None], mean, np.nan)
        return Targets(pooled.astype(np.float16)).standardised()

    @staticmethod
    def write(root: DataRoot, source: str) -> Path:
        """`source`'s targets of every trained run, sentence-pooled, under `source` + `SUFFIX`; those there kept."""
        origin, directory = Targets.directory(root, source), Targets.directory(root, source + SUFFIX)
        for run in DataConfig(val=Val.OFFICIAL).trained(Run.discover(root.raw())):
            path, kept = origin / f"{run.name}.npy", directory / f"{run.name}.npy"
            if path.exists() and not kept.exists():
                Sentences(RunCache.words(run)[Column.SENTENCE].to_numpy()).pool(Targets(np.load(path))).save(kept)
                LOG.info("%s", kept.name)
        return directory

    @staticmethod
    def main() -> None:
        """Pool the word-level targets named on the command line into sentence-level ones."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("source", help="a directory of word-level targets under processed/speech")
        LOG.info("wrote %s", Sentences.write(DataRoot.from_config(), parser.parse_args().source))


if __name__ == "__main__":
    Sentences.main()

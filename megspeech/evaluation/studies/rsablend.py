"""Which teacher's 50-word geometry matches what a trained ensemble confuses, read from its saved probabilities
(megspeech-7gf.141.1).

The MEG matrix is the members' mean probability of each of the 50 candidates for every true word, over the unseen-span
words of the `official-val` section of each member's `probs.pt`, symmetrised. A teacher's matrix is the cosine of its
type-level word vectors over subject 0's training words. A word-frequency control is the matrix of log-count gaps
between two words' training counts. Spearman over the upper triangle, a bootstrap over words for the interval. CPU
only; no model is run.

    CUDA_VISIBLE_DEVICES= uv run --no-sync python -m megspeech.evaluation.studies.rsablend m11wd
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import DEEP, Column
from megspeech.device import Device
from megspeech.evaluation.group import Group
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities
from megspeech.evaluation.studies.rsa import DRAWS, HEADER, Meg, Row, Rsa, Study

LOG = logging.getLogger(__name__)

SECTION = "official-val"
METRICS = "metrics.json"


@dataclass(frozen=True)
class Frequency:
    """How often each vocabulary word is a training label: the control a teacher must beat to say more than 'common
    words are confused with common words'."""

    counts: Int[np.ndarray, " v"]

    @classmethod
    def of(cls, words: pd.DataFrame) -> "Frequency":
        labels = words[Column.TARGET].to_numpy()
        return cls(np.bincount(labels[(labels >= 0) & (labels < VOCAB_SIZE)], minlength=VOCAB_SIZE))

    @property
    def present(self) -> Bool[np.ndarray, " v"]:
        return self.counts > 0

    def similarity(self) -> Float[np.ndarray, "v v"]:
        """Minus the gap between two words' log counts: nearer in frequency is more alike."""
        log = np.log(np.maximum(self.counts, 1))
        return -np.abs(log[:, None] - log[None, :])

    def rsa(self, meg: Meg, draws: int = DRAWS) -> Rsa:
        return Rsa.of(meg.similarity(), self.similarity(), meg.present & self.present, draws)

    def markdown(self, meg: Meg, draws: int = DRAWS) -> str:
        rsa = self.rsa(meg, draws)
        return (
            f"| word frequency | 1 | {rsa.words} | {rsa.rho:+.3f} | [{rsa.low:+.3f}, {rsa.high:+.3f}] "
            f"| {rsa.p:.3f} | - | - | - | - | - |"
        )


@dataclass(frozen=True)
class Blend:
    """An ensemble's saved probabilities for one section, the members averaged."""

    members: tuple[str, ...]
    section: str

    @classmethod
    def named(cls, models: Path, name: str, section: str = SECTION) -> "Blend":
        listed = json.loads((models / name / METRICS).read_text(encoding="utf-8"))["members"]
        return cls(tuple(listed), section)

    def meg(self, models: Path) -> Meg:
        """The members' mean probabilities on the unseen-span words of every domain, rowed by true word."""
        total: np.ndarray | None = None
        labels: np.ndarray | None = None
        for member in self.members:
            domains = Probabilities.load(models / member / PROBS, self.section).domains
            scored = [domains[key] for key in sorted(domains)]
            probabilities = np.concatenate([words.probabilities[words.novel].numpy() for words in scored])
            truth = np.concatenate([words.labels[words.novel].numpy() for words in scored])
            if labels is not None and not np.array_equal(labels, truth):
                raise ValueError(f"{member}: scored words differ from the first member's")
            labels = truth
            total = probabilities if total is None else total + probabilities
        if total is None or labels is None:
            raise ValueError("no members")
        return Meg.of(total / len(self.members), labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="an ensemble under models/ whose metrics.json lists members with probs.pt")
    parser.add_argument("--section", default=SECTION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    blend = Blend.named(root.models(), args.name, args.section)
    meg = blend.meg(root.models())
    LOG.info(
        "%s: %d members, MEG matrix %d of %d words, %d scored",
        args.name,
        len(blend.members),
        int(meg.present.sum()),
        VOCAB_SIZE,
        int(meg.counts.sum()),
    )
    study = Study(root, Loader.of(Group(root, Scoring(), Device.CPU.as_torch()), [blend.members[0]]))
    LOG.info(HEADER)
    LOG.info("|---|---|---|---|---|---|---|---|---|---|---|")
    LOG.info(Frequency.of(study.training(DEEP)).markdown(meg))
    for teacher, geometry in study.teachers(DEEP).items():
        LOG.info(Row.of(teacher, geometry, meg).markdown())


if __name__ == "__main__":
    main()

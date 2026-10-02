"""Turn per-window vocabulary probabilities into a Deep-track Kaggle submission.

The first predictor is the training-set word-frequency prior: one constant row for every holdout window. Under
BAcc@10 a constant row ranks the same ten words first for every class, so it must score exactly chance (20%) —
the floor that proves the CSV, the vocabulary order and the scorer agree before any model is spent.

    uv run python -m megspeech.submission.submission --out prior.csv [--submit "message"]
"""

import argparse
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from pnpl.competition import PRIMARY_VOCAB, SECONDARY_VOCAB, submit_to_kaggle, write_submission
from pnpl.competition.holdout import LibriBrainCompetitionHoldout
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.continuous import MISSING
from megspeech.data.holdout import TRACK, HoldoutCache
from megspeech.data.split import Split
from megspeech.data.vocabulary import SECONDARY, Vocabulary
from megspeech.evaluation.report import Evaluator, Head

LOG = logging.getLogger(__name__)

COMPETITION = "pnpl-competition-2026-deep"


@dataclass(frozen=True)
class FrequencyPrior:
    """How often each vocabulary word is heard in the training events, smoothed so no word is impossible."""

    counts: Counter[str]

    @classmethod
    def from_events(cls, paths: Iterable[Path]) -> "FrequencyPrior":
        """Count the `word` rows of LibriBrain events files, spelled the way the competition vocabulary is."""
        counts: Counter[str] = Counter()
        for path in paths:
            events = pd.read_csv(path, sep="\t", usecols=["kind", "segment"], keep_default_na=False, na_values=MISSING)
            words = events.loc[events["kind"] == "word", "segment"].dropna()
            counts.update(Vocabulary.spell(word) for word in words)
        return cls(counts)

    def probabilities(self, vocabulary: list[str]) -> Float[np.ndarray, " v"]:
        """Add-one-smoothed relative frequency over `vocabulary`, in its order."""
        counts = np.array([self.counts[word] + 1 for word in vocabulary], dtype=np.float64)
        return counts / counts.sum()


class Rows(Protocol):
    """What a submission needs of the holdout: how many windows, and their canonical indices."""

    indices: np.ndarray

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class Submission:
    """The Deep holdout's canonical rows and how to fill them."""

    holdout: Rows

    def write(self, path: Path, primary: Float[np.ndarray, "n v"], secondary: Float[np.ndarray, "n w"]) -> Path:
        """One probability row per holdout window, in canonical order."""
        return write_submission(path, self.holdout.indices, primary, secondary)

    def write_constant(self, path: Path, primary: Float[np.ndarray, " v"], secondary: Float[np.ndarray, " w"]) -> Path:
        """The same probability rows for every holdout window."""
        rows = len(self.holdout)
        return self.write(path, np.tile(primary, (rows, 1)), np.tile(secondary, (rows, 1)))


@dataclass(frozen=True)
class Submitter:
    """A model's holdout predictions, read through the same evaluator that scored it on validation."""

    evaluator: Evaluator

    def submit(self, model: nn.Module, root: DataRoot, log_prior: Float[Tensor, " v"], tau: float, path: Path) -> Path:
        """Context-head probabilities for every holdout row, logit-adjusted by `tau`, written as a submission."""
        holdout = LibriBrainCompetitionHoldout(TRACK, data_path=root.raw(), download=False)
        cache = HoldoutCache.of(root)
        cache.build()
        split = Split.from_holdout(holdout, cache.windows(self.evaluator.device))
        logits = self.evaluator.logits(model, split)[Head.CONTEXT]
        primary = torch.softmax(logits - tau * log_prior, dim=-1).cpu().numpy()
        secondary = np.tile(FrequencyPrior(Counter()).probabilities(list(SECONDARY.words)), (len(primary), 1))
        return Submission(holdout).write(path, primary, secondary)


def main() -> None:
    """Write the frequency-prior submission and optionally upload it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", metavar="MESSAGE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raw = DataRoot.from_config().raw()
    events = [path for path in sorted(raw.rglob("*sub-0_*_events.tsv")) if path.stat().st_size]
    prior = FrequencyPrior.from_events(events)
    LOG.info("prior from %d heard words", sum(prior.counts.values()))
    holdout = LibriBrainCompetitionHoldout(TRACK, data_path=raw, download=False)
    csv = Submission(holdout).write_constant(
        args.out, prior.probabilities(PRIMARY_VOCAB), prior.probabilities(SECONDARY_VOCAB)
    )
    if args.submit:
        submit_to_kaggle(csv, competition=COMPETITION, message=args.submit)


if __name__ == "__main__":
    main()

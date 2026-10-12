"""The frequency-prior submission for a track (Deep or Broad).

The first predictor is the training-set word-frequency prior: one constant row for every Kaggle test window. Under
BAcc@10 a constant row ranks the same ten words first for every class, so it must score exactly chance (20%) —
the floor that proves the CSV, the vocabulary order and the scorer agree before any model is spent.

    uv run python -m megspeech.submission.prior --out prior.csv [--track broad] [--submit "message"]
"""

import argparse
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from pnpl.competition import PRIMARY_VOCAB, SECONDARY_VOCAB, submit_to_kaggle, write_submission

from core.paths import DataRoot
from megspeech.data.cache.kaggle import Track
from megspeech.data.constants import MISSING
from megspeech.data.labels.vocabulary import Vocabulary
from megspeech.data.raw import Event, Kind

LOG = logging.getLogger(__name__)

COMPETITION = "pnpl-competition-2026-{track}"


@dataclass(frozen=True)
class FrequencyPrior:
    """How often each vocabulary word is heard in the training events, smoothed so no word is impossible."""

    counts: Counter[str]

    @classmethod
    def from_events(cls, paths: Iterable[Path]) -> "FrequencyPrior":
        """Count the `word` rows of LibriBrain events files, spelled the way the competition vocabulary is."""
        counts: Counter[str] = Counter()
        for path in paths:
            events = pd.read_csv(
                path, sep="\t", usecols=[Event.KIND, Event.SEGMENT], keep_default_na=False, na_values=MISSING
            )
            words = events.loc[events[Event.KIND] == Kind.WORD, Event.SEGMENT].dropna()
            counts.update(Vocabulary.spell(word) for word in words)
        return cls(counts)

    def probabilities(self, vocabulary: list[str]) -> Float[np.ndarray, " v"]:
        """Add-one-smoothed relative frequency over `vocabulary`, in its order."""
        counts = np.array([self.counts[word] + 1 for word in vocabulary], dtype=np.float64)
        return counts / counts.sum()


def main() -> None:
    """Write the frequency-prior submission and optionally upload it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--track", type=Track, choices=list(Track), default=Track.DEEP)
    parser.add_argument("--submit", metavar="MESSAGE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raw = DataRoot.from_config().raw()
    events = [path for path in sorted(raw.rglob("*sub-0_*_events.tsv")) if path.stat().st_size]
    prior = FrequencyPrior.from_events(events)
    LOG.info("prior from %d heard words", sum(prior.counts.values()))
    kaggle = args.track.kaggle(raw)
    primary, secondary = prior.probabilities(PRIMARY_VOCAB), prior.probabilities(SECONDARY_VOCAB)
    rows = (len(kaggle), 1)
    csv = write_submission(args.out, kaggle.indices, np.tile(primary, rows), np.tile(secondary, rows))
    if args.submit:
        submit_to_kaggle(csv, competition=COMPETITION.format(track=args.track), message=args.submit)


if __name__ == "__main__":
    main()

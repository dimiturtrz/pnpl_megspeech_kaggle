"""Whether other subjects heard subject 0's stimuli: the ruler's Sherlock1 text in their pooled runs, and the
Kaggle test's sentences in their archives.

A run's words are matched by `Text`; if two runs are one recording of one text, the onsets of matched words differ
by one constant shift. A Kaggle test sentence is keyed by its word-onset vector: two archives holding one sentence hold
the same onsets.

    uv run python -m megspeech.evaluation.studies.overlap
"""

import argparse
import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float
from pnpl.competition.holdout import N_SUBJECTS

from core.paths import DataRoot
from megspeech.data.cache.kaggle import ArchiveKind, Epochs, KaggleCache
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.labels.vocabulary import ABSENT, PRIMARY
from megspeech.data.raw.runs import DEEP, Partition, Run
from megspeech.evaluation.text import RULER, Text

LOG = logging.getLogger(__name__)

TOLERANCE = 1e-3
"""Seconds two Kaggle test onsets may differ by and still be one word; the archives store onsets in 10 ms steps."""


@dataclass(frozen=True)
class Overlap:
    """How much of a reference run another subject's run repeats: matched words of `total`, the vocabulary words
    among them, the reference sentences matched whole, and the matched onsets' median shift and widest departure
    from it in seconds."""

    subject: int
    words: int
    total: int
    labelled: int
    sentences: int
    shift: float
    spread: float

    @classmethod
    def of(cls, reference: Text, other: Text, subject: int) -> "Overlap":
        at, words = reference.matches(other), reference.words
        found = at >= 0
        shift = other.words["onset"].to_numpy()[at[found]] - words["onset"].to_numpy()[found]
        median = float(np.median(shift)) if found.any() else float("nan")
        labelled = PRIMARY.labels(words["word"].astype(str)) != ABSENT
        whole = pd.Series(found).groupby(words["sentence"].to_numpy()).all()
        return cls(
            subject,
            int(found.sum()),
            len(found),
            int((found & labelled).sum()),
            int(whole.sum()),
            median,
            float(np.abs(shift - median).max(initial=0.0)),
        )


@dataclass(frozen=True)
class SentenceMatch:
    """How many of a reference archive's sentences another subject's archive holds by onset vector: matched at all,
    and matched to exactly one sentence that no other reference sentence also matches; and over the one-to-one pairs
    the widest difference in epoch length, in samples."""

    subject: int
    matched: int
    unique: int
    length_spread: int
    total: int


@dataclass(frozen=True)
class Onsets:
    """A sentence archive's word onsets, `nan` past each sentence's last word."""

    epochs: Epochs

    @property
    def onset(self) -> Float[np.ndarray, "e w"]:
        return np.where(self.epochs.mask, self.epochs.onset, np.nan)

    def equal(self, other: "Onsets", tolerance: float = TOLERANCE) -> Bool[np.ndarray, "e f"]:
        """Whether each of these sentences has every onset of each of `other`'s, and the same words missing."""
        a, b = self.onset[:, None], other.onset[None]
        same = (np.abs(a - b) <= tolerance) | (np.isnan(a) & np.isnan(b))
        return same.all(axis=2)

    def match(self, other: "Onsets", subject: int) -> SentenceMatch:
        equal = self.equal(other)
        unique = (equal.sum(axis=1) == 1) & (equal & (equal.sum(axis=0) == 1)).any(axis=1)
        partner = equal.argmax(axis=1)
        spread = np.abs(self.epochs.length - other.epochs.length[partner])[unique].max(initial=0)
        return SentenceMatch(subject, int(equal.any(axis=1).sum()), int(unique.sum()), int(spread), len(equal))


@dataclass(frozen=True)
class Survey:
    """The overlap check over a data root: subject 0's ruler run against every pooled run, and subject 0's Kaggle test
    sentences against every subject's."""

    root: DataRoot

    def text(self) -> list[Overlap]:
        runs = Run.discover(self.root.raw())
        reference = next(
            r for r in runs if r.subject == DEEP and r.partition is Partition.VALIDATION and r.key[0] == RULER
        )
        pooled = Run.pooled(runs)
        table = RunCache.under(self.root.processed()).published()
        texts = {run.name: Text.of(table[table["run"] == run.name]) for run in [reference, *pooled]}
        return [Overlap.of(texts[reference.name], texts[run.name], run.subject) for run in pooled]

    def kaggle(self) -> list[SentenceMatch]:
        cache = replace(KaggleCache.of(self.root), subjects=tuple(range(N_SUBJECTS)))
        sentences = [Onsets(a.epochs()) for a in cache.archives() if a.name.endswith(ArchiveKind.SENTENCE)]
        onsets = dict(zip(cache.subjects, sentences, strict=True))
        return [onsets[DEEP].match(other, subject) for subject, other in onsets.items()]


def main() -> None:
    """Log, per subject, how much of subject 0's ruler text its pooled run repeats and how many of subject 0's
    Kaggle test sentences its Kaggle test archive holds, with the totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    survey = Survey(DataRoot.from_config())
    text = survey.text()
    for row in text:
        LOG.info("text %s", row)
    LOG.info(
        "text: %d runs, %d of %d words matched (%d vocabulary)",
        len(text),
        sum(r.words for r in text),
        sum(r.total for r in text),
        sum(r.labelled for r in text),
    )
    matches = survey.kaggle()
    for match in matches:
        LOG.info("Kaggle test %s", match)
    others = [match for match in matches if match.subject != DEEP]
    LOG.info(
        "Kaggle test: %d of %d subject-0 sentences matched one-to-one across %d other subjects",
        sum(e.unique for e in others),
        sum(e.total for e in others),
        len(others),
    )


if __name__ == "__main__":
    main()

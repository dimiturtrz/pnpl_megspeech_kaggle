import numpy as np
import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring.probabilities import Words
from megspeech.evaluation.studies.signal.overdispersion import (
    ABSENT,
    Cases,
    Dispersion,
    Outcome,
    Overdispersion,
    Ranks,
    Spans,
)
from tests.unit.megspeech.evaluation.scoring.evaluator import split

LABELS = 5
SPANS = 40
PER_SPAN = 6
PERMS = 200
ALPHA = 0.05
FLAGGED_AT_MOST = 0.25
DETECTED_AT_LEAST = 0.75
LOW_MEAN, HIGH_MEAN = 0.25, 0.75


def cases(offset: float, difficulty: float, seed: int) -> Cases:
    """Spans of `PER_SPAN` words, each word's score its label's difficulty plus noise plus its span's offset."""
    generator = torch.Generator().manual_seed(seed)
    span = torch.arange(SPANS).repeat_interleave(PER_SPAN)
    stratum = torch.randint(LABELS, (len(span),), generator=generator)
    shift = torch.randn(SPANS, generator=generator)[span] * offset
    score = torch.randn(LABELS, generator=generator)[stratum] * difficulty + torch.randn(len(span), generator=generator)
    score = score + shift
    return Cases(stratum, span, Ranks(score, (score > 0).float()))


def pvalues(offset: float, difficulty: float, measure: str, runs: int = 12) -> np.ndarray:
    return np.array([cases(offset, difficulty, seed).dispersion(measure).test(PERMS, seed).p for seed in range(runs)])


def test_ranks_of() -> None:
    probabilities = torch.zeros(2, VOCAB_SIZE)
    probabilities[0, 0] = 1.0
    probabilities[1, :12] = 1.0
    probabilities[1, 20] = 0.5
    ranks = Ranks.of(probabilities, torch.tensor([0, 20]))
    assert ranks.score.tolist() == pytest.approx([1.0, 1 - 12 / (VOCAB_SIZE - 1)])
    assert ranks.hit.tolist() == [1.0, 0.0]


def test_residual() -> None:
    values = torch.tensor([1.0, 3.0, 5.0, 10.0])
    stratum = torch.tensor([0, 0, 0, 1])
    out = Cases.residual(values, stratum)
    assert out[:3].tolist() == pytest.approx([1 - 4.0, 3 - 3.0, 5 - 2.0])
    assert out[3].isnan()


def test_measure() -> None:
    sample = cases(0.0, 0.0, 0)
    assert torch.equal(sample.measure("hit"), sample.ranks.hit)
    assert torch.equal(sample.measure("score"), sample.ranks.score)


def test_spans() -> None:
    assert cases(0.0, 1.0, 0).dispersion("score").spans == SPANS
    assert Dispersion(torch.zeros(0), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)).spans == 0


def test_dispersion() -> None:
    sample = cases(0.0, 1.0, 0)
    out = sample.dispersion("score")
    assert out.spans == SPANS
    assert (out.stratum.diff() >= 0).all()
    gone = Cases(sample.stratum, torch.full_like(sample.span, ABSENT), sample.ranks)
    assert len(gone.dispersion("score").residual) == 0


def test_icc() -> None:
    clustered = cases(2.0, 0.0, 0).dispersion("score")
    flat = cases(0.0, 0.0, 0).dispersion("score")
    assert float(clustered.icc(clustered.span[None])[0]) > 0.3
    assert abs(float(flat.icc(flat.span[None])[0])) < 0.15


def test_shuffled() -> None:
    sample = cases(0.0, 1.0, 0).dispersion("score")
    out = sample.shuffled(3, torch.Generator().manual_seed(0))
    assert out.shape == (3, len(sample.span))
    for row in out:
        for label in sample.stratum.unique():
            kept = sample.stratum == label
            assert torch.equal(row[kept].sort().values, sample.span[kept].sort().values)


def test_test() -> None:
    sample = cases(0.0, 1.0, 0).dispersion("score")
    outcome = sample.test(PERMS, seed=1)
    assert outcome.low <= outcome.null_mean <= outcome.high
    assert outcome.spans == SPANS
    assert outcome == sample.test(PERMS, seed=1)


def test_no_sentence_effect_is_not_significant() -> None:
    p = pvalues(0.0, 0.0, "score")
    assert (p < ALPHA).mean() <= FLAGGED_AT_MOST
    assert LOW_MEAN < p.mean() < HIGH_MEAN


def test_sentence_offset_is_significant() -> None:
    assert pvalues(0.7, 0.0, "score").max() < 0.01
    assert (pvalues(0.7, 0.0, "hit") < ALPHA).mean() >= DETECTED_AT_LEAST


def test_label_difficulty_alone_is_not_flagged() -> None:
    score, hit = pvalues(0.0, 2.0, "score"), pvalues(0.0, 2.0, "hit")
    assert (score < ALPHA).mean() <= FLAGGED_AT_MOST
    assert LOW_MEAN < score.mean() < HIGH_MEAN
    assert (hit < ALPHA).mean() <= FLAGGED_AT_MOST


def test_spans_of() -> None:
    domain = split(n=12)
    spans = Spans.of(domain)
    assert len(spans.of_word) == 12
    assert (spans.of_word >= 0).all()
    assert len(spans.of_word.unique()) == len(domain.index)


def test_spans_read() -> None:
    spans = Spans.of(split(n=12))
    assert spans.read(torch.tensor([0, 1])).tolist() == spans.of_word[:2].tolist()


def words(n: int, seed: int = 0) -> Words:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, VOCAB_SIZE, generator=generator)
    labels = torch.arange(n) % 3
    return Words(torch.arange(n), labels, torch.ones(n, dtype=torch.bool), logits, logits, logits.softmax(-1))


def test_overdispersion_of() -> None:
    study = Overdispersion.of({"d": words(12)}, {"d": split(n=12)})
    assert list(study.domains) == ["d"]
    assert len(study.domains["d"].span) == 12


def test_pooled() -> None:
    study = Overdispersion.of({"a": words(12), "b": words(12, 1)}, {"a": split(n=12), "b": split(n=12)})
    pooled = study.pooled()
    assert len(pooled.span) == 24
    assert len(pooled.span.unique()) == 2 * len(study.domains["a"].span.unique())
    assert len(pooled.stratum.unique()) == 6


def test_report() -> None:
    study = Overdispersion.of({"d": words(12)}, {"d": split(n=12)})
    lines = study.report(20)
    assert len(lines) == 4
    assert lines[0].lstrip().startswith("d score")


def test_line() -> None:
    assert "n_words=5" in Outcome(0.1, 0.0, -0.1, 0.1, 0.5, 5, 2).line("d", "score")

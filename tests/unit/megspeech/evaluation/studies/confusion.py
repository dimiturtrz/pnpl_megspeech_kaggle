from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.confusion import Collected, Gains, Heatmap, Lengths, Report, Scored, Structure

PAIR = 2


def scored() -> Scored:
    """Even targets peak on themselves; odd ones peak on their even neighbour, true word second."""
    labels = np.arange(2 * VOCAB_SIZE) % VOCAB_SIZE
    probs = np.full((len(labels), VOCAB_SIZE), 0.001)
    for row, label in enumerate(labels):
        probs[row, label - label % PAIR] = 0.5
        probs[row, label] = max(probs[row, label], 0.01)
    return Scored(labels, probs, np.where(np.arange(len(labels)) < VOCAB_SIZE, "A", "B"))


def paired() -> Structure:
    """Only targets 0 and 1 are confused, three times each way; every other word is read right."""
    counts = np.eye(VOCAB_SIZE, dtype=np.int64)
    counts[0, 1] = counts[1, 0] = 3
    return Structure(counts)


def test_top1() -> None:
    assert scored().top1()[:4].tolist() == [0, 0, 2, 2]


def test_hits() -> None:
    assert scored().hits().all()


def test_support() -> None:
    assert (scored().support() == PAIR).all()


def test_confusion() -> None:
    counts = scored().confusion()
    assert counts[0, 0] == PAIR
    assert counts[1, 0] == PAIR
    assert counts[1, 1] == 0


def test_recall() -> None:
    recall = scored().recall()
    assert recall[0::PAIR].tolist() == [1.0] * (VOCAB_SIZE // PAIR)
    assert recall[1::PAIR].tolist() == [0.0] * (VOCAB_SIZE // PAIR)


def test_recall_absent_target() -> None:
    one = Scored(np.array([0]), np.eye(VOCAB_SIZE)[:1], np.array(["A"]))
    assert np.isnan(one.recall()[1])


def test_hit_rate() -> None:
    assert scored().hit_rate().tolist() == [1.0] * VOCAB_SIZE


def test_macro() -> None:
    assert scored().macro() == {"A": 1.0, "B": 1.0, "macro": 1.0}


def test_rates() -> None:
    rates = paired().rates()
    assert rates[0].tolist()[:2] == pytest.approx([0.25, 0.75])
    assert rates.sum(axis=1) == pytest.approx(np.ones(VOCAB_SIZE))


def test_affinity() -> None:
    affinity = paired().affinity()
    assert affinity[0, 1] == affinity[1, 0] == pytest.approx(0.75)
    assert np.diag(affinity).sum() == 0


def test_tree() -> None:
    assert paired().tree().shape == (VOCAB_SIZE - 1, 4)


def test_order() -> None:
    assert sorted(paired().order().tolist()) == list(range(VOCAB_SIZE))


def test_pairs() -> None:
    assert paired().pairs(1) == [(0, 1, 3, 3)]


def test_clusters() -> None:
    assert paired().clusters() == [[0, 1]]


def test_pair_values() -> None:
    values = paired().pair_values()
    assert len(values) == VOCAB_SIZE * (VOCAB_SIZE - 1) // PAIR
    assert values.max() == pytest.approx(0.75)


def test_letter_counts() -> None:
    assert Lengths.letter_counts(["a", "it’s"]).tolist() == [1.0, 3.0]


def test_phoneme_counts(tmp_path: Path) -> None:
    """Counts phonemes per aligned word, skipping `oov`/`sil` positions; an unseen word is NaN."""
    events = tmp_path / "events.tsv"
    events.write_text(
        "kind\tsegment\tsentenceidx\twordidx\n"
        "word\tThe\t0\t0\nphoneme\tdh_B\t0\t0\nphoneme\tah_I\t0\t0\nphoneme\tsil_I\t0\t0\n"
    )
    counts = Lengths.phoneme_counts([events], ["the", "zzz"])
    assert counts[0] == PAIR
    assert np.isnan(counts[1])


def test_durations() -> None:
    split = cast(
        "Split",
        SimpleNamespace(
            index=torch.tensor([[0, 1, 2], [3, PAD, PAD]]),
            onset=torch.tensor([[0.0, 1.0, 3.0], [5.0, 0.0, 0.0]]),
            y=torch.arange(4),
        ),
    )
    gap = Lengths.durations(split, torch.arange(4))
    assert gap[:2].tolist() == [1.0, 2.0]
    assert np.isnan(gap[2:]).all()


def test_mean_by_target() -> None:
    mean = Lengths.mean_by_target(np.array([0, 0, 1]), np.array([1.0, 3.0, np.nan]))
    assert mean[0] == PAIR
    assert np.isnan(mean[1])


def test_spearman() -> None:
    """Confusion that falls as size gap grows is a rank correlation of -1 on every measure."""
    size = np.arange(VOCAB_SIZE, dtype=np.float64)
    rows, cols = np.triu_indices(VOCAB_SIZE, 1)
    out = Lengths(size, size, size).spearman(-np.abs(size[rows] - size[cols]))
    assert out["letters"][0] == pytest.approx(-1.0)
    assert out["duration"][2] == len(rows)


def test_heatmap_save(tmp_path: Path) -> None:
    rates = paired().rates()
    path = Heatmap(PRIMARY.words, paired().order()).save(tmp_path / "out" / "map.png", {"a": rates, "b": rates})
    assert path.stat().st_size > 0


def test_gains_of() -> None:
    gains = Gains.of(np.zeros(VOCAB_SIZE), paired())
    assert np.flatnonzero(gains.clustered).tolist() == [0, 1]


def test_concentration() -> None:
    gain = np.arange(VOCAB_SIZE) / VOCAB_SIZE
    out = Gains.of(gain, paired()).concentration()
    assert (out["n-inside"], out["n-outside"]) == (PAIR, VOCAB_SIZE - PAIR)
    assert out["mean-inside"] == pytest.approx(gain[:PAIR].mean())
    assert out["share-of-total-inside"] == pytest.approx(gain[:PAIR].sum() / gain.sum())


def test_run(tmp_path: Path) -> None:
    """The whole report runs on a synthetic pair of models and writes its artefacts."""
    size = np.arange(VOCAB_SIZE, dtype=np.float64)
    collected = Collected(scored(), scored(), np.ones(2 * VOCAB_SIZE))
    Report(collected, PRIMARY.words, Lengths(size, size, size), tmp_path).run()
    assert np.load(tmp_path / "timing_confusion.npy").shape == (VOCAB_SIZE, VOCAB_SIZE)
    assert (tmp_path / "confusion.png").exists()

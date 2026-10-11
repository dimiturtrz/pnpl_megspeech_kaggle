import numpy as np

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.scored import Scored

PAIR = 2


def scored() -> Scored:
    """Even targets peak on themselves; odd ones peak on their even neighbour, true word second."""
    labels = np.arange(2 * VOCAB_SIZE) % VOCAB_SIZE
    probs = np.full((len(labels), VOCAB_SIZE), 0.001)
    for row, label in enumerate(labels):
        probs[row, label - label % PAIR] = 0.5
        probs[row, label] = max(probs[row, label], 0.01)
    return Scored(labels, probs, np.where(np.arange(len(labels)) < VOCAB_SIZE, "A", "B"))


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

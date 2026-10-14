import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.objective.auxiliary.contrastive.similarity import Similarity
from megspeech.objective.auxiliary.contrastive.word import SameWord, SameWordTerm
from megspeech.objective.term import Reading

CPU = torch.device("cpu")
WORDS = pd.DataFrame({"run": ["a"] * 5, "position": range(5), "sentence": [0, 0, 1, 1, 2]})
LABEL = torch.tensor([7, 3, 7, IGNORE, 7])
"""Rows 0 and 1 share sentence 0, rows 2 and 3 sentence 1; rows 0, 2, 4 are one word, row 3 is outside the 50."""
SENTENCE = torch.tensor([0, 0, 1, 1, 2])


def packed() -> torch.Tensor:
    return SameWord.pack(LABEL, SENTENCE)


def test_pack() -> None:
    assert packed().tolist() == [7, 3, VOCAB_SIZE + 7, IGNORE, 2 * VOCAB_SIZE + 7]


def test_pairs() -> None:
    """Same label from another sentence is a positive; same-sentence pairs and unlabelled rows are no candidate;
    other labels from other sentences are negatives; the slot itself is none."""
    candidate, positive = SameWord().pairs(packed())
    assert {(int(i), int(j)) for i, j in positive.nonzero()} == {(0, 2), (2, 0), (0, 4), (4, 0), (2, 4), (4, 2)}
    assert not candidate[0, 1]
    assert not candidate[1, 0]
    assert candidate[1, 2]
    assert not positive[1, 2]
    assert not candidate[3].any()
    assert not candidate[:, 3].any()
    assert not candidate.diagonal().any()


def test_pairs_same_sentence_same_label() -> None:
    candidate, positive = SameWord().pairs(SameWord.pack(torch.tensor([7, 7]), torch.tensor([0, 0])))
    assert not candidate.any()
    assert not positive.any()


def test_of() -> None:
    made = SameWordTerm.of(0.5, WORDS, CPU)
    assert made.weight == 0.5
    assert made.sentence.tolist() == [0, 0, 1, 1, 2]
    assert not SameWordTerm.local


def test_positives() -> None:
    assert isinstance(SameWordTerm.of(1.0, WORDS, CPU).positives(), SameWord)


def test_labels() -> None:
    split = Split.from_arrays(np.zeros((5, 4, 4), np.float32), LABEL.numpy(), Spans.by_gap(np.arange(5) * 2.0), CPU)
    assert torch.equal(SameWordTerm.of(1.0, WORDS, CPU).labels(split), packed())
    with pytest.raises(ValueError, match="sentence rows"):
        SameWordTerm.of(1.0, WORDS.iloc[:4], CPU).labels(split)


def test_unweighted() -> None:
    """Moving same-word embeddings from different sentences together lowers the loss."""
    term = SameWordTerm.of(1.0, WORDS, CPU)
    target = packed()[None]
    rows = torch.ones(target.shape)
    spread = torch.eye(5, 6)[None] * 2
    pulled = spread.clone()
    pulled[0, 2], pulled[0, 4] = spread[0, 0], spread[0, 0]
    assert term.unweighted(Reading(Similarity(), pulled, target, rows)) < term.unweighted(
        Reading(Similarity(), spread, target, rows)
    )

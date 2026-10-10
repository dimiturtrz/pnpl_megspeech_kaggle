import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities, Words
from megspeech.evaluation.studies.rsa import Meg
from megspeech.evaluation.studies.rsablend import SECTION, Blend, Frequency

DRAWS = 50
PER_WORD = 3
MEMBERS = ("a", "b")


def frequency() -> Frequency:
    counts = np.arange(1, VOCAB_SIZE + 1)
    return Frequency(counts)


def words(offset: float) -> Words:
    labels = torch.arange(VOCAB_SIZE).repeat(PER_WORD)
    n = len(labels)
    probabilities = torch.full((n, VOCAB_SIZE), 0.1 / (VOCAB_SIZE - 1))
    probabilities[torch.arange(n), labels] = 0.9 - offset
    probabilities[torch.arange(n), (labels + 1) % VOCAB_SIZE] += offset
    novel = torch.ones(n, dtype=torch.bool)
    novel[0] = False
    zeros = torch.zeros(n, VOCAB_SIZE)
    return Words(torch.arange(n), labels, novel, zeros, zeros, probabilities)


def write_ensemble(models: Path, offsets: tuple[float, float]) -> None:
    (models / "ens").mkdir()
    (models / "ens" / "metrics.json").write_text(json.dumps({"members": list(MEMBERS)}), encoding="utf-8")
    for member, offset in zip(MEMBERS, offsets, strict=True):
        (models / member).mkdir()
        probabilities = Probabilities({"dom": words(offset)}, 1.0, torch.zeros(VOCAB_SIZE), 1)
        probabilities.save(models / member / PROBS, SECTION)


def test_frequency_of() -> None:
    table = pd.DataFrame({"target": [0, 0, 3, -1, VOCAB_SIZE]})
    counts = Frequency.of(table).counts
    assert counts.shape == (VOCAB_SIZE,)
    assert counts[0] == 2
    assert counts[3] == 1
    assert counts.sum() == 3


def test_present() -> None:
    present = Frequency(np.array([0, 2, 5])).present
    assert present.tolist() == [False, True, True]


def test_similarity() -> None:
    similarity = frequency().similarity()
    assert np.allclose(np.diag(similarity), 0.0)
    assert similarity[0, 1] == pytest.approx(-np.log(2))
    assert np.allclose(similarity, similarity.T)


def test_rsa() -> None:
    first = frequency()
    meg = Meg(first.similarity(), np.ones(VOCAB_SIZE, dtype=int))
    rsa = first.rsa(meg, DRAWS)
    assert rsa.rho == pytest.approx(1.0)
    assert rsa.words == VOCAB_SIZE


def test_markdown() -> None:
    first = frequency()
    meg = Meg(first.similarity(), np.ones(VOCAB_SIZE, dtype=int))
    line = first.markdown(meg, DRAWS)
    assert line.startswith("| word frequency | 1 | 50 | +1.000")


def test_named(tmp_path: Path) -> None:
    write_ensemble(tmp_path, (0.0, 0.0))
    blend = Blend.named(tmp_path, "ens")
    assert blend.members == MEMBERS
    assert blend.section == SECTION


def test_meg(tmp_path: Path) -> None:
    write_ensemble(tmp_path, (0.0, 0.2))
    meg = Blend.named(tmp_path, "ens").meg(tmp_path)
    assert meg.counts.sum() == VOCAB_SIZE * PER_WORD - 1
    assert meg.counts[0] == PER_WORD - 1
    assert meg.scores[5, 5] == pytest.approx(0.8)
    assert meg.scores[5, 6] == pytest.approx(0.1 / (VOCAB_SIZE - 1) + 0.1)

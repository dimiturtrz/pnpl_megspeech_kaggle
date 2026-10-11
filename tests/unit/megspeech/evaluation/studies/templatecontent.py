import pandas as pd
import pytest
import torch

from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary
from megspeech.evaluation.studies.templatecontent import Durations, Rsa


def test_durations_add() -> None:
    durations = Durations()
    durations.add(torch.tensor([0, 0, 1]), torch.tensor([1.0, 3.0, float("nan")]))
    assert durations.counts[:2].tolist() == [2.0, 0.0]


def test_mean() -> None:
    durations = Durations()
    durations.add(torch.tensor([0, 0]), torch.tensor([1.0, 3.0]))
    assert durations.mean()[0] == 2.0


def test_gap() -> None:
    assert Rsa.gap(torch.tensor([0.0, 1.0, 4.0]))[0].tolist() == [0.0, 1.0, 4.0]


def test_phonetic() -> None:
    table = Pronunciation(
        torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words))).sort().values,
        torch.zeros(VOCAB_SIZE, 4, dtype=torch.long),
    )
    assert Rsa.phonetic(table).sum() == 0


def test_semantic() -> None:
    distance = Rsa.semantic(torch.eye(3))
    assert distance.diagonal().abs().max() < 1e-6
    assert distance[0, 1] == pytest.approx(1.0)


def test_ranks() -> None:
    assert Rsa.ranks(torch.tensor([3.0, 1.0, 1.0, 2.0])).tolist() == [4.0, 1.5, 1.5, 3.0]


def test_correlate() -> None:
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(12, 3, generator=generator)
    target = torch.cdist(points, points)
    words = torch.ones(12, dtype=torch.bool)
    rsa = Rsa(permutations=200)
    r, p, pairs = rsa.correlate(target + 0.01 * torch.rand(12, 12), target, words)
    assert r > 0.9
    assert p < 0.05
    assert pairs == 66
    assert abs(rsa.correlate(torch.rand(12, 12), target, words)[0]) < 0.5

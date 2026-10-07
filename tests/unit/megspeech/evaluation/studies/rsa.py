from pathlib import Path

import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.rsa import GemmaWords, Geometry, Meg, Row, Rsa

DIM = 6
DRAWS = 50


def random_geometry(seed: int = 0, dim: int = DIM) -> Geometry:
    return Geometry(np.random.default_rng(seed).normal(size=(VOCAB_SIZE, dim)))


def test_present() -> None:
    vectors = random_geometry().vectors.copy()
    vectors[3] = 0
    present = Geometry(vectors).present
    assert not present[3]
    assert present.sum() == VOCAB_SIZE - 1


def test_cosine() -> None:
    vectors = random_geometry().vectors.copy()
    vectors[3] = 0
    similarity = Geometry(vectors).cosine()
    assert np.allclose(np.diag(similarity)[np.arange(VOCAB_SIZE) != 3], 1.0)
    assert np.isnan(similarity[3]).all()
    assert np.isnan(similarity[:, 3]).all()


def test_min_distance() -> None:
    assert Geometry.one_hot().min_distance() == 1.0
    vectors = random_geometry().vectors.copy()
    vectors[1] = vectors[0] * 2
    assert abs(Geometry(vectors).min_distance()) < 1e-9


def test_truncated() -> None:
    head = random_geometry().truncated(3)
    assert head.vectors.shape == (VOCAB_SIZE, 3)
    assert np.allclose(np.linalg.norm(head.vectors, axis=1), 1.0)


def test_one_hot() -> None:
    assert np.array_equal(Geometry.one_hot().cosine(), np.eye(VOCAB_SIZE))


def test_meg_of() -> None:
    probabilities = np.zeros((4, VOCAB_SIZE))
    probabilities[:, 0] = [1.0, 0.5, 0.0, 0.2]
    meg = Meg.of(probabilities, np.array([0, 0, 1, 1]))
    assert meg.counts[:3].tolist() == [2, 2, 0]
    assert np.isclose(meg.scores[0, 0], 0.75)
    assert np.isclose(meg.scores[1, 0], 0.1)
    assert (meg.scores[2] == 0).all()


def test_meg_present() -> None:
    meg = Meg.of(np.full((2, VOCAB_SIZE), 1 / VOCAB_SIZE), np.array([4, 7]))
    assert np.flatnonzero(meg.present).tolist() == [4, 7]


def test_meg_similarity() -> None:
    scores = np.random.default_rng(0).random((VOCAB_SIZE, VOCAB_SIZE))
    similarity = Meg(scores, np.ones(VOCAB_SIZE, dtype=int)).similarity()
    assert np.allclose(similarity, similarity.T)
    assert np.isclose(similarity[1, 2], (scores[1, 2] + scores[2, 1]) / 2)


def test_spearman() -> None:
    line = np.arange(10.0)
    assert np.isclose(Rsa.spearman(line, line**3), 1.0)
    assert np.isclose(Rsa.spearman(line, -line), -1.0)
    assert Rsa.spearman(line, np.ones(10)) == 0.0


def test_pairs() -> None:
    similarity = random_geometry().cosine()
    rows = np.arange(VOCAB_SIZE)
    assert np.isclose(Rsa.pairs(similarity, similarity, rows), 1.0)
    assert np.isclose(Rsa.pairs(similarity, similarity, np.array([3, 3, 5, 7, 9])), 1.0)


def test_of() -> None:
    similarity = random_geometry().cosine()
    noise = random_geometry(1).cosine()
    shared = np.ones(VOCAB_SIZE, dtype=bool)
    same = Rsa.of(similarity, similarity, shared, DRAWS)
    assert same.rho == 1.0
    assert same.low > 0.99
    assert same.words == VOCAB_SIZE
    apart = Rsa.of(similarity, noise, shared, DRAWS)
    assert apart.low < apart.rho < apart.high
    assert abs(apart.rho) < 0.2


def test_markdown() -> None:
    meg = Meg(random_geometry().cosine(), np.ones(VOCAB_SIZE, dtype=int))
    line = Row.of("teacher", random_geometry(), meg, DRAWS).markdown()
    assert line.startswith(f"| teacher | {DIM} | {VOCAB_SIZE} | ")
    assert line.count("|") == len(line.split("|")) - 1 == 7


def test_row_of() -> None:
    meg = Meg(random_geometry().cosine(), np.ones(VOCAB_SIZE, dtype=int))
    row = Row.of("self", random_geometry(), meg, DRAWS)
    assert row.rsa.rho == 1.0
    null = Row.of("one-hot", Geometry.one_hot(), meg, DRAWS)
    assert null.rsa.rho == 0.0
    assert (null.rsa.low, null.rsa.high) == (0.0, 0.0)


def test_pooled() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [0.0, 9.0]]])
    pooled = GemmaWords.pooled(hidden, torch.tensor([[1, 1, 0]]))
    assert torch.allclose(pooled, torch.tensor([[1.0, 0.0]]))


def test_path(tmp_path: Path) -> None:
    assert GemmaWords.path(DataRoot(tmp_path)).suffix == ".npy"


def test_save(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert GemmaWords(random_geometry().vectors).save(root) == GemmaWords.path(root)
    assert GemmaWords.path(root).exists()


def test_load(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    vectors = random_geometry().vectors
    GemmaWords(vectors).save(root)
    assert np.array_equal(GemmaWords.load(root).vectors, vectors)

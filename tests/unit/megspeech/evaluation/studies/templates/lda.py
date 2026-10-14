import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.templates.lda import Dataset, Harness, Lda, Outcome, Pca

CLASSES = 4


def classes(n: int) -> torch.Tensor:
    return torch.arange(n) % CLASSES


def separable(n: int = 400, width: int = 5) -> Dataset:
    y = classes(n)
    x = torch.randn(n, width, generator=torch.Generator().manual_seed(0)) * 0.1
    x[:, 0] += y.float()
    return Dataset(x, y, torch.arange(n) // 10)


def test_take() -> None:
    data = separable()
    assert len(data.take(data.group == 0).y) == 10


def test_pca_fit() -> None:
    pca = Pca.fit(separable().x)
    assert pca.axes.shape == (5, 5)
    assert torch.allclose(pca.axes.T @ pca.axes, torch.eye(5), atol=1e-4)


def test_project() -> None:
    data = separable()
    z = Pca.fit(data.x).project(data.x, 2)
    assert z.shape == (len(data.y), 2)
    assert torch.allclose(z.mean(dim=0), torch.zeros(2), atol=1e-4)


def test_lda_fit() -> None:
    lda = Lda.fit(separable().x, separable().y)
    assert lda.present.sum() == CLASSES
    assert lda.means.shape == (VOCAB_SIZE, 5)


def test_scores() -> None:
    data = separable()
    scores = Lda.fit(data.x, data.y).scores(data.x)
    assert (scores.argmax(dim=1) == data.y).float().mean() > 0.95
    assert scores[:, CLASSES:].max() < scores[:, :CLASSES].min()


def test_macro() -> None:
    assert Outcome(3, {"a": 0.2, "b": 0.4}).macro == pytest.approx(0.3)


def test_dims() -> None:
    assert Harness().dims(100) == [5, 10, 20, 34, 52, 63, 79, 89, 98]


def test_balanced() -> None:
    y = classes(40)
    scores = torch.nn.functional.one_hot(y, VOCAB_SIZE).float()
    assert Harness.balanced(scores, y) == 1.0


def test_decoded() -> None:
    data = separable()
    assert Harness().decoded(data, data, 3, Pca.fit(data.x)) > 0.9


def test_inner() -> None:
    data = separable()
    assert 1 <= Harness().inner(data) <= 5


def test_outcome() -> None:
    data = separable()
    outcome = Harness().outcome(data, {"d": data})
    assert outcome.domains["d"] > 0.9

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from megspeech.data.batching.split import EVAL_SPANS
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.viz.models.embed import COMPONENTS, PROJECTORS, Embedder, Manifold, Principal, Projection
from megspeech.model.sentence import SentenceNet
from tests.unit.megspeech.features.trained import WORDS, split
from tests.unit.megspeech.model.sentence import CONTEXT

CPU = torch.device("cpu")


def net() -> SentenceNet:
    torch.manual_seed(0)
    return SentenceNet(nn.Identity(), CONTEXT)


def test_captured() -> None:
    model = nn.ModuleList([net(), net()]).eval()
    words = split()
    inputs = words.batch(torch.arange(len(words.index)), EVAL_SPANS * words.index.shape[1]).inputs()
    taken: list[Tensor] = []
    with Embedder.captured(model, taken), torch.no_grad():
        for member in model:
            member(inputs)
    assert len(taken) == 2
    assert taken[0].shape[0] == taken[1].shape[0]
    with torch.no_grad():
        model[0](inputs)
    assert len(taken) == 2


def test_embedded() -> None:
    model = net()
    embedded = Embedder(Evaluator(CPU)).embedded(model, split())
    assert embedded.shape[0] == WORDS
    assert embedded.shape[1] > 0
    assert torch.isfinite(embedded).all()
    assert not torch.equal(embedded[0], embedded[1])


def test_flat() -> None:
    projection = Projection.flat(4)
    assert projection.points.shape == (4, COMPONENTS)
    assert not projection.points.any()
    assert projection.explained.shape == (0,)


def test_principal_project() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4)) * np.array([5.0, 2.0, 0.1, 0.1])
    projection = Principal().project(x)
    assert projection.points.shape == (50, COMPONENTS)
    assert projection.explained[0] > projection.explained[1] > 0
    assert projection.explained.sum() <= 1
    assert np.abs(projection.points[:, 0]).std() > np.abs(projection.points[:, 1]).std()
    assert Principal().project(np.zeros((3, 1))).explained.tolist() == pytest.approx([0, 0])


def blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    centre = np.zeros(16)
    centre[0] = 10.0
    return np.concatenate([rng.normal(size=(60, 16)) + centre, rng.normal(size=(60, 16)) - centre])


def test_manifold_project() -> None:
    x = blobs()
    projection = Manifold().project(x)
    assert projection.points.shape == (len(x), COMPONENTS)
    assert projection.explained.shape == (0,)
    assert np.array_equal(projection.points, Manifold().project(x).points)
    first, second = projection.points[:60], projection.points[60:]
    between = np.linalg.norm(first.mean(axis=0) - second.mean(axis=0))
    within = max(
        np.linalg.norm(first - first.mean(axis=0), axis=1).max(),
        np.linalg.norm(second - second.mean(axis=0), axis=1).max(),
    )
    assert between > within


def test_manifold_project_degenerate() -> None:
    for x in (np.zeros((30, 0)), np.ones((5, 4)), np.ones((30, 4))):
        projection = Manifold().project(x)
        assert projection.points.shape == (len(x), COMPONENTS)
        assert not projection.points.any()


def test_projectors() -> None:
    assert [p.name for p in PROJECTORS] == ["pca", "umap"]

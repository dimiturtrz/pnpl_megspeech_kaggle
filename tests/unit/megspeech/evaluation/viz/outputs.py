import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.metric.ruler import TOP, Graded
from megspeech.evaluation.viz.outputs import METRIC, Located, Outputs, Predictions, Reader

RUN = "sub-0_ses-11_task-Sherlock1_run-2"
PROJECTIONS = {"pca": {"explained": [0.2, 0.1]}, "umap": {"explained": []}}


def outputs() -> Outputs:
    return Outputs("model", str(Track.DEEP), 0.5, PROJECTIONS, {"0000": {"pred": "the"}}, {"mrr": 0.5})


def test_manifest() -> None:
    assert outputs().manifest() == {
        "name": "model",
        "track": str(Track.DEEP),
        "labels": list(PRIMARY.words),
        "metric": METRIC,
        "score": 0.5,
        "metrics": {"mrr": 0.5},
    }


def test_write(tmp_path: Path) -> None:
    path = outputs().write(tmp_path / "models")
    assert path == tmp_path / "models" / "model.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "name": "model",
        "projections": PROJECTIONS,
        "samples": {"0000": {"pred": "the"}},
    }


def test_records() -> None:
    probabilities = torch.full((2, VOCAB_SIZE), 0.001)
    probabilities[0, 3] = 0.9
    probabilities[1, 5] = 0.5
    labels = torch.tensor([3, VOCAB_SIZE - 1])
    points = {"pca": np.array([[1.0, 2.0], [3.0, 4.0]]), "umap": np.array([[5.0, 6.0], [7.0, 8.0]])}
    records = Predictions().records(["a", "b"], probabilities, labels, points)
    assert records["a"]["pred"] == PRIMARY.words[3]
    assert records["a"]["correct"] is True
    assert records["a"]["p_true"] == pytest.approx(0.9)
    assert records["a"]["error"] == pytest.approx(0.1)
    assert records["b"]["correct"] is False
    assert records["b"]["pred"] == PRIMARY.words[5]
    assert len(cast("list[object]", records["b"]["topk"])) == TOP
    assert records["b"]["emb"] == {"pca": [3.0, 4.0], "umap": [7.0, 8.0]}
    assert records["a"]["rank"] == 1
    rank = records["b"]["rank"]
    assert isinstance(rank, int)
    assert rank > TOP


def test_rank() -> None:
    probabilities = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]])
    assert Predictions.rank(probabilities, torch.tensor([0, 1, 0])).tolist() == [1, 2, 3]


def test_metrics() -> None:
    probabilities = torch.full((4, VOCAB_SIZE), 0.001)
    probabilities[0, 0] = 0.9
    probabilities[1, 0] = 0.8
    probabilities[1, 1] = 0.5
    probabilities[2, 1] = 0.9
    probabilities[3, 0] = 0.7
    labels = torch.tensor([0, 1, 1, 2])
    metrics = Predictions().metrics(probabilities, labels)
    assert metrics["bacc@1"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert metrics["f1@1"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert metrics[f"bacc@{TOP}"] == pytest.approx(2 / 3)
    assert 0 < metrics["map"] <= 1
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 + 1 / VOCAB_SIZE) / 4)


def rows() -> dict[str, pd.DataFrame]:
    return {
        "a": pd.DataFrame({"run": [RUN] * 3, "onset": [1.0, 2.0, 3.0]}),
        "b": pd.DataFrame({"run": [RUN] * 2, "onset": [4.0, 5.0]}),
    }


def located() -> Located:
    keys = pd.DataFrame({"run": [RUN] * 3, "onset": [5.0, 2.0, 3.0], "id": ["0", "1", "2"]})
    return Located.of(keys, rows())


def test_of() -> None:
    found = located()
    assert list(found.domain) == ["b", "a", "a"]
    assert found.position.tolist() == [1, 1, 2]
    with pytest.raises(KeyError):
        Located.of(pd.DataFrame({"run": [RUN], "onset": [9.0], "id": ["0"]}), rows())


def test_at() -> None:
    assert located().at("a").tolist() == [1, 2]
    assert located().at("c").tolist() == []


def test_sampled() -> None:
    graded = Graded(
        {
            "a": (torch.tensor([True, False, True]), torch.tensor([1, 2, 3])),
            "b": (torch.tensor([True]), torch.tensor([4])),
        }
    )
    kept = located().sampled(graded, {"a": torch.tensor([0, 1, 2]), "b": torch.tensor([0])})
    assert list(kept.domains) == ["a"]
    hits, labels = kept.domains["a"]
    assert hits.tolist() == [False, True]
    assert labels.tolist() == [2, 3]


def test_tau(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    saved = root.models() / "model" / METRICS
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"best": {"macro": {"tau": 0.25}}}), encoding="utf-8")
    assert Reader.tau(root, "model") == 0.25


def loaded(subjects: tuple[int, ...]) -> ExportedModel:
    return cast("ExportedModel", SimpleNamespace(data=SimpleNamespace(subjects=subjects)))


def test_track() -> None:
    assert Reader.track(loaded((0,))) == str(Track.DEEP)
    assert Reader.track(loaded((0, 1))) == str(Track.BROAD)

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.config import METRICS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation import loading as module
from megspeech.evaluation import reader as reader_module
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.model.input.subjects import Reading
from tests.unit.megspeech.evaluation.loaded import CPU, Favourite


def members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selections: dict[str, int]) -> None:
    """Trained runs named by `selections`, each recording its `min_novel`, behind a stubbed word table and model."""
    heard = pd.DataFrame(
        {"run": "r", "sentence": 0, "onset": [0.0, 0.1], "target": [0, 1], "token": [7, 8], "weight": 1.0}
    )
    monkeypatch.setattr(reader_module.Corpus, "heard", classmethod(lambda *_: heard))
    monkeypatch.setattr(reader_module.Run, "discover", classmethod(lambda *_: []))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: Favourite()))
    for name, min_novel in selections.items():
        directory = DataRoot(tmp_path).models() / name
        directory.mkdir(parents=True)
        config: dict[str, dict[str, int]] = {"data": {}, "selection": {"min_novel": min_novel}}
        (directory / METRICS).write_text(json.dumps({"config": config}))


def test_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Members are scored by the prior training drew and the selection that counted them, or not at all."""
    members(tmp_path, monkeypatch, {"a": 5, "b": 5, "c": 7})
    (DataRoot(tmp_path).models() / "ab").mkdir()
    (DataRoot(tmp_path).models() / "ab" / METRICS).write_text(json.dumps({"members": ["a", "b"]}))
    scoring = Scoring(balance=Sinkhorn(2), shifts=(0, 1), reading=Reading.UNSEEN)
    built = Loader.of(DataRoot(tmp_path), ["ab"], scoring, CPU)
    expected = torch.tensor([1.0, 1.0] + [0.0] * (VOCAB_SIZE - 2)) + 1
    assert torch.allclose(built.log_prior, (expected / expected.sum()).log())
    assert built.evaluator.seen == frozenset({(7, 8)})
    assert (built.evaluator.min_novel, built.evaluator.balance, built.evaluator.shifts) == (5, Sinkhorn(2), (0, 1))
    assert (built.names, built.reading) == (("a", "b"), Reading.UNSEEN)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        Loader.of(DataRoot(tmp_path), ["b", "c"], scoring, CPU)

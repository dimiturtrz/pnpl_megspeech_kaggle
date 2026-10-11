from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import AS_RECORDED
from megspeech.evaluation.group import Group
from megspeech.evaluation.scoring.config import Scoring
from tests.unit.megspeech.evaluation.loaded import CPU
from tests.unit.megspeech.evaluation.loading import members


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runs of one reading load as one model that remembers their names and the selection that counted them."""
    members(tmp_path, monkeypatch, {"a": 5, "b": 5})
    directories = [DataRoot(tmp_path).models() / name for name in ("a", "b")]
    loaded = Group(DataRoot(tmp_path), Scoring(), CPU, AS_RECORDED).load(directories)
    assert loaded.names == ("a", "b")
    assert loaded.evaluator.min_novel == 5
    assert loaded.evaluator.seen == frozenset({(7, 8)})
    assert torch.isclose(loaded.log_prior.exp().sum(), torch.tensor(1.0))

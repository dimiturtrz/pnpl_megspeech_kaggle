import logging
from pathlib import Path

import numpy as np
import pytest

from megspeech.data.cache.kaggle import Track
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.labelweighted import Shrinkage
from megspeech.submission.labelweighted import LabelWeighted
from megspeech.submission.report import SUBMISSIONS
from tests.unit.megspeech.evaluation.labelweighted import TEST_ROWS, pool


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """One normalised row per Kaggle test row is written under out, and the in-sample score and entropy are logged."""
    rows: list[np.ndarray] = []

    def write(_: Track, __: Path, primary: np.ndarray, path: Path) -> Path:
        rows.append(primary)
        return path

    monkeypatch.setattr(Track, "write", write)
    with caplog.at_level(logging.INFO, logger="megspeech.submission.labelweighted"):
        path = LabelWeighted(pool(kaggle=True), Shrinkage(5.0, 10.0), Track.BROAD).write(tmp_path, tmp_path / "out")
    assert path == tmp_path / "out" / SUBMISSIONS[Track.BROAD]
    assert rows[0].shape == (TEST_ROWS, VOCAB_SIZE)
    assert np.allclose(rows[0].sum(axis=1), 1.0, atol=1e-5)
    assert "in-sample weighted ruler" in caplog.text
    assert "weight entropy" in caplog.text

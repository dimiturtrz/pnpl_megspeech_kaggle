import logging
from pathlib import Path

import pytest

from megspeech.evaluation.loaded import RULER
from megspeech.evaluation.rulers import deep as module
from megspeech.evaluation.rulers.deep import DeepRuler
from tests.unit.megspeech.evaluation.loaded import loaded, runs


def test_select(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=module.__name__):
        report = DeepRuler.select(loaded(tmp_path), runs(tmp_path))
    assert RULER in report.domains
    assert f"macro {report.macro.bacc10:.4f} tau {report.tau} over {report.macro.n} words" in caplog.text


def test_graded(tmp_path: Path) -> None:
    graded = DeepRuler.graded(loaded(tmp_path), runs(tmp_path), 0.5)
    assert RULER in graded.domains

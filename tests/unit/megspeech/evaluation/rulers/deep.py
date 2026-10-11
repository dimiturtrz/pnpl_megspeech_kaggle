import logging
from pathlib import Path

import pytest

from megspeech.evaluation.rulers import deep as module
from megspeech.evaluation.rulers.deep import DeepRuler
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded import loaded, runs


def test_select(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=module.__name__):
        report = DeepRuler(loaded(tmp_path), runs(tmp_path)).select()
    assert RULER in report.domains
    assert f"macro {report.macro.bacc10:.4f} tau {report.tau} over {report.macro.n} words" in caplog.text
    novel = [domain.novel for domain in report.domains.values()]
    mean = sum(s.bacc10 for s in novel) / len(novel)
    assert f"novel-text macro {mean:.4f} over {sum(s.n for s in novel)} words" in caplog.text


def test_graded(tmp_path: Path) -> None:
    graded = DeepRuler(loaded(tmp_path), runs(tmp_path)).graded(0.5)
    assert RULER in graded.domains

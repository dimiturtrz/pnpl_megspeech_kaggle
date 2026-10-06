import logging
from pathlib import Path

import pytest

from megspeech.data.raw.runs import Partition
from megspeech.evaluation.rulers import localtest as module
from megspeech.evaluation.rulers.localtest import LocalTest
from tests.unit.megspeech.evaluation.loaded import loaded, runs


def local(tmp_path: Path) -> LocalTest:
    return LocalTest(loaded(tmp_path), runs(tmp_path))


def test_tau(tmp_path: Path) -> None:
    assert local(tmp_path).tau() == loaded(tmp_path).report(runs(tmp_path)).tau


def test_graded(tmp_path: Path) -> None:
    """Test scores subject 0's official test run alone; validation scores the validation domains."""
    study = local(tmp_path)
    assert list(study.graded(Partition.TEST, study.tau()).domains) == ["Sherlock1"]
    assert len(study.graded(Partition.VALIDATION, study.tau()).domains) == 2


def test_summary(tmp_path: Path) -> None:
    study = local(tmp_path)
    graded = study.graded(Partition.TEST, study.tau())
    assert f"macro {study.ruler.point(graded):.4f} [" in study.summary(graded)
    assert "Sherlock1" in study.summary(graded)


def test_report(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=module.__name__):
        local(tmp_path).report("label")
    assert [r.getMessage().split(" tau ")[0] for r in caplog.records] == ["label val", "label local test"]

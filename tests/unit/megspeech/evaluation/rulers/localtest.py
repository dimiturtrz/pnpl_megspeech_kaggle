import logging
from pathlib import Path

import pytest

from megspeech.data.raw.runs import Partition
from megspeech.evaluation.rulers import localtest as module
from megspeech.evaluation.rulers.localtest import LocalTest
from megspeech.evaluation.scoring.probabilities import PROBS, SECTIONS, Probabilities
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


def test_record(tmp_path: Path) -> None:
    """A partition's per-word probabilities are saved under its own section, at the tau asked."""
    study = local(tmp_path)
    path = study.record(Partition.TEST, 0.5, tmp_path / PROBS)
    saved = Probabilities.load(path, SECTIONS[Partition.TEST])
    assert list(saved.domains) == ["Sherlock1"]
    assert saved.tau == 0.5
    study.record(Partition.VALIDATION, 0.5, path)
    assert len(Probabilities.load(path, SECTIONS[Partition.VALIDATION]).domains) == 2
    assert list(Probabilities.load(path, SECTIONS[Partition.TEST]).domains) == ["Sherlock1"]


def test_report_saved(tmp_path: Path) -> None:
    local(tmp_path).report("label", tmp_path / PROBS)
    sections = {key.split("/")[0].split("#")[0] for key in Probabilities.stored(tmp_path / PROBS)}
    assert sections == set(SECTIONS.values())

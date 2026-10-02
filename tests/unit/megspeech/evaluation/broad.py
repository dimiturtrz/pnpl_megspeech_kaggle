import logging
import sys
from pathlib import Path

import pytest
import torch
from pnpl.competition.holdout import N_SUBJECTS

from core.paths import DataRoot
from megspeech.evaluation import broad as module
from megspeech.evaluation.broad import BroadRuler, Options, Reading, main
from megspeech.evaluation.report import Score
from megspeech.model.subjects import Unseen
from tests.unit.megspeech.evaluation.pooling import CPU, SPOKEN, pooling, runs

POOLED = 2


def test_reading_subjects() -> None:
    subject = torch.tensor([1, 5, 32])
    assert torch.equal(Reading.OWN.subjects(subject), subject)
    assert Reading.UNSEEN.subjects(subject).tolist() == [N_SUBJECTS] * 3


def test_broad_ruler_split(tmp_path: Path) -> None:
    every = runs(tmp_path)
    own = BroadRuler(pooling(tmp_path)).split(every[POOLED])
    assert own.subject.tolist() == [1] * len(SPOKEN)
    unseen = BroadRuler(pooling(tmp_path), Reading.UNSEEN).split(every[POOLED])
    assert unseen.subject.tolist() == [N_SUBJECTS] * len(SPOKEN)


def test_broad_ruler_score(tmp_path: Path) -> None:
    """Every word of the pooled run is novel to a model that saw no orders; the reversed ranking hits none."""
    score = BroadRuler(pooling(tmp_path)).score(runs(tmp_path)[POOLED])
    assert score.n == len(SPOKEN)
    assert score.bacc10 == pytest.approx(0.0)


def test_broad_ruler_scores(tmp_path: Path) -> None:
    scores = BroadRuler(pooling(tmp_path)).scores(runs(tmp_path))
    assert list(scores) == [1]
    assert scores[1].n == len(SPOKEN)


def test_broad_ruler_report(caplog: pytest.LogCaptureFixture) -> None:
    scores = {1: Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 5), 2: Score.best({0.0: 0.6, 0.5: 0.4, 1.0: 0.1}, 7)}
    with caplog.at_level(logging.INFO, logger=module.__name__):
        macro = BroadRuler.report(scores)
    assert (macro.tau, macro.n) == (0.0, 12)
    assert macro.bacc10 == pytest.approx(0.4)
    assert "subject  1  bacc10 0.2000  n 5" in caplog.text
    assert "over 2 subjects, 12 words" in caplog.text


def test_options_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["broad", "m", "--reading", "unseen", "--unseen", "mean"])
    options = Options.parse(None)
    assert (options.name, options.reading, options.unseen) == ("m", Reading.UNSEEN, Unseen.MEAN)
    monkeypatch.setattr(sys, "argv", ["broad", "m"])
    assert (Options.parse(None).reading, Options.parse(None).unseen) == (Reading.OWN, Unseen.DEEP)


def test_options_ruler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(module.Pooling, "of", classmethod(lambda _, *args: calls.append(args) or pooling(tmp_path)))
    options = Options("m", module.DataConfig().grouping, Unseen.DEEP, Reading.UNSEEN)
    ruler = options.ruler(DataRoot(tmp_path), CPU)
    assert ruler.reading is Reading.UNSEEN
    assert calls[0][-1] is Unseen.DEEP


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    every = runs(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(module.Pooling, "of", classmethod(lambda *_: pooling(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["broad", "m"])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        main()
    assert f"over 1 subjects, {len(SPOKEN)} words" in caplog.text

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from megspeech.data.batching.split import BATCH, EVAL_SPANS
from megspeech.data.config import METRICS
from megspeech.evaluation.metric.ruler import Interval
from megspeech.evaluation.scoring.score import (
    ADJUSTMENTS,
    MIN_NOVEL,
    UNSELECTED,
    Epoch,
    Forward,
    Score,
    SelectionConfig,
)


def test_best() -> None:
    score = Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7)
    assert (score.tau, score.bacc10, score.n) == (0.5, 0.4, 7)
    assert Score.best({0.0: math.nan, 0.5: 0.4, 1.0: 0.3}, 7).tau == 0.5
    assert Score.best(dict.fromkeys(ADJUSTMENTS, math.nan), 7).tau == UNSELECTED


def test_mean() -> None:
    """One adjustment is chosen for the unweighted mean, not each score's own best."""
    scores = [Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7), Score.best({0.0: 0.6, 0.5: 0.4, 1.0: 0.1}, 3)]
    macro = Score.mean(iter(scores))
    assert (macro.tau, macro.n) == (0.0, 10)
    assert macro.bacc10 == pytest.approx(0.4)
    assert macro.by_tau == pytest.approx({0.0: 0.4, 0.5: 0.4, 1.0: 0.2})


def test_mean_empty() -> None:
    """A macro over no counted domain scores nothing rather than a number."""
    macro = Score.mean([])
    assert macro.n == 0
    assert all(math.isnan(value) for value in macro.by_tau.values())


def test_average() -> None:
    assert Score.average(iter([0.2, 0.4])) == pytest.approx(0.3)
    assert math.isnan(Score.average([]))


def selected(directory: Path, config: dict[str, object]) -> Path:
    directory.mkdir()
    (directory / METRICS).write_text(json.dumps({"config": config}))
    return directory


def test_saved(tmp_path: Path) -> None:
    assert SelectionConfig.saved(selected(tmp_path / "old", {})) == SelectionConfig(MIN_NOVEL)
    assert SelectionConfig.saved(selected(tmp_path / "new", {"selection": {"min_novel": 7}})).min_novel == 7


def test_shared(tmp_path: Path) -> None:
    """Members scored alike share their selection; members that counted domains differently cannot be pooled."""
    one = selected(tmp_path / "one", {"selection": {"min_novel": 7}})
    two = selected(tmp_path / "two", {"selection": {"min_novel": 7}})
    other = selected(tmp_path / "other", {})
    assert SelectionConfig.shared([one, two]) == SelectionConfig(7)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        SelectionConfig.shared([one, other])


def test_tau() -> None:
    macro = Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7)
    assert Epoch(0, 0.0, {}, macro, Interval(0.3, 0.5), 0.1, 0.0).tau == 0.5


def test_steering() -> None:
    macro = Score.best({0.0: 0.2, 0.5: 0.4, 1.0: 0.3}, 7)
    row = replace(Epoch(0, 0.0, {}, macro, Interval(0.3, 0.5), 0.1, 0.0), fit=0.9, shuffled=0.25)
    steering = row.steering()
    assert steering["gap"] == pytest.approx(0.5)
    assert steering["neural"] == pytest.approx(0.15)
    bare = replace(row, fit=None, shuffled=None).steering()
    assert bare["gap"] is None
    assert bare["neural"] is None


def test_spans() -> None:
    assert Forward().spans() == EVAL_SPANS
    assert Forward(BATCH // 8).spans() == EVAL_SPANS // 8
    assert Forward(1).spans() >= 1


def test_fewest(tmp_path: Path) -> None:
    """Runs share the smallest batch any of them trained at; a record without an optim section trained at `BATCH`."""
    configs: dict[str, dict[str, dict[str, int]]] = {"small": {"optim": {"batch": BATCH // 8}}, "default": {}}
    for name, config in configs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / METRICS).write_text(json.dumps({"config": config}))
    assert Forward.fewest([tmp_path / "default"]) == Forward()
    assert Forward.fewest([tmp_path / "default", tmp_path / "small"]) == Forward(BATCH // 8)

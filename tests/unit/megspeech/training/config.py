import dataclasses
import json
from pathlib import Path

from megspeech.data.config import DataConfig
from megspeech.evaluation.metric.ruler import Interval
from megspeech.evaluation.scoring.score import Epoch, Score
from megspeech.model.context.timing import Clock
from megspeech.training.config import METRICS, RunConfig
from megspeech.training.regularise.augment import Augment


def test_saved(tmp_path: Path) -> None:
    flat = {"epochs": 3, "read_weight": 4.0, "min_novel": 7, "augment": {"dropout": 0.1, "gain": 0.0}}
    (tmp_path / METRICS).write_text(json.dumps({"config": {**flat, "context": {"clock": "order"}}}))
    config = RunConfig.saved(tmp_path)
    assert config.optim.epochs == 3
    assert config.data.read_weight == 4.0
    assert config.selection.min_novel == 7
    assert config.augment == Augment(0.1, 0.0)
    assert config.context.clock is Clock.ORDER
    (tmp_path / METRICS).write_text(json.dumps({"config": dataclasses.asdict(config)}))
    assert RunConfig.saved(tmp_path) == config


def epoch(at: int, bacc10: float) -> Epoch:
    return Epoch(at, 0.0, {}, Score(0.5, bacc10, {0.5: bacc10}, 1), Interval(bacc10, bacc10), 0.0, 0.0)


def test_selected() -> None:
    history = [epoch(0, 0.2), epoch(1, 0.4), epoch(2, 0.4), epoch(3, 0.3)]
    assert RunConfig().selected(history) is history[1]
    assert RunConfig(data=DataConfig(refit=True)).selected(history) is history[-1]

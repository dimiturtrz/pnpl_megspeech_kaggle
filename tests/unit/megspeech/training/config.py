import dataclasses
import json
from pathlib import Path

import pytest

from megspeech.data.config import DataConfig
from megspeech.evaluation.metric.ruler import Interval
from megspeech.evaluation.scoring.score import Epoch, Score
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.context.timing import Clock
from megspeech.model.words.conv import ConvConfig
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


@pytest.mark.parametrize("context", [ContextConfig(encoder=Encoder.CACHED), ContextConfig(conv=ConvConfig(merger=270))])
def test_subspace_needs_sensor_free_encoder(context: ContextConfig) -> None:
    assert RunConfig(context=context).context is context
    with pytest.raises(ValueError, match="306 sensors"):
        RunConfig(data=DataConfig(subspace=True), context=context)
    assert RunConfig(data=DataConfig(subspace=True)).data.subspace

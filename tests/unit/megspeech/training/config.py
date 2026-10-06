import dataclasses
import json
from pathlib import Path

import pytest

from megspeech.data.config import DataConfig, Refit, Val
from megspeech.evaluation.metric.ruler import Interval
from megspeech.evaluation.scoring.score import Epoch, Score
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.context.timing import Clock
from megspeech.model.words.conv import ConvConfig
from megspeech.training.config import METRICS, OptimConfig, RunConfig, StopConfig
from megspeech.training.regularise.augment import Augment
from megspeech.training.sampling.curriculum import Base


def test_saved_retired_data_fields(tmp_path: Path) -> None:
    """A record with the retired `extra`/`share`/`stratify`/`ruler` or boolean `refit` loads under `val`/`refit`."""
    (tmp_path / METRICS).write_text(json.dumps({"config": {"data": {"extra": 0, "read_weight": 2.0}}}))
    legacy = RunConfig.saved(tmp_path).data
    assert (legacy.read_weight, legacy.val, legacy.refit) == (2.0, Val.OFFICIAL, Refit.NONE)
    old = {"ruler": "balanced", "refit": True}
    (tmp_path / METRICS).write_text(json.dumps({"config": {"data": old}}))
    balanced = RunConfig.saved(tmp_path).data
    assert (balanced.val, balanced.refit) == (Val.CARVED, Refit.VAL)


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
    assert RunConfig(data=DataConfig(refit=Refit.VAL)).selected(history) is history[-1]
    assert RunConfig(optim=OptimConfig(heads=0.0)).selected(history) is history[-1]
    assert RunConfig(optim=OptimConfig(last=True)).selected(history) is history[-1]


@pytest.mark.parametrize("context", [ContextConfig(encoder=Encoder.CACHED), ContextConfig(conv=ConvConfig(merger=270))])
def test_projected_needs_sensor_free_encoder(context: ContextConfig) -> None:
    assert RunConfig(context=context).context is context
    for data in (DataConfig(subspace=True), DataConfig(regions=("Left-temporal",))):
        with pytest.raises(ValueError, match="306 sensors"):
            RunConfig(data=data, context=context)
        assert RunConfig(data=data).data.projected


def test_saved_flat_optim_sampler(tmp_path: Path) -> None:
    """A run saved with the sampler knobs under `optim` loads them into `sampler`."""
    old = {"optim": {"epochs": 3, "priority": 0.5, "sampler": "uniform", "stratified": 2.0}}
    (tmp_path / METRICS).write_text(json.dumps({"config": old}))
    config = RunConfig.saved(tmp_path)
    assert config.optim.epochs == 3
    assert (config.sampler.priority, config.sampler.base, config.sampler.alpha) == (0.5, Base.UNIFORM, 2.0)


def test_stalled() -> None:
    rising = [epoch(i, 0.1 * i) for i in range(6)]
    peaked = [epoch(0, 0.2), epoch(1, 0.5), epoch(2, 0.5), epoch(3, 0.4), epoch(4, 0.3)]
    assert not StopConfig().stalled(peaked)
    assert not StopConfig(3).stalled(rising)
    assert not StopConfig(4).stalled(peaked)
    assert StopConfig(3).stalled(peaked)
    assert StopConfig(1).stalled(peaked[:3])


def test_keeps_last() -> None:
    assert not RunConfig().keeps_last
    assert RunConfig(data=DataConfig(refit=Refit.VAL)).keeps_last
    assert RunConfig(optim=OptimConfig(last=True)).keeps_last
    assert RunConfig(optim=OptimConfig(heads=0.0)).keeps_last


def test_post_init_patience() -> None:
    assert RunConfig(stop=StopConfig(8)).stop.patience == 8
    with pytest.raises(ValueError, match="selected on validation"):
        RunConfig(optim=OptimConfig(heads=0.0), stop=StopConfig(8))
    with pytest.raises(ValueError, match="selected on validation"):
        RunConfig(data=DataConfig(refit=Refit.VAL), stop=StopConfig(8))
    with pytest.raises(ValueError, match="negative"):
        RunConfig(stop=StopConfig(-1))


def test_saved_without_stop(tmp_path: Path) -> None:
    (tmp_path / METRICS).write_text(json.dumps({"config": {"optim": {"epochs": 3}}}))
    assert RunConfig.saved(tmp_path).stop == StopConfig()
    (tmp_path / METRICS).write_text(json.dumps({"config": {"stop": {"patience": 5}}}))
    assert RunConfig.saved(tmp_path).stop.patience == 5

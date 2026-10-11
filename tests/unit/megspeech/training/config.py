import dataclasses
import json
from pathlib import Path

import pytest

from megspeech.data.config import DataConfig, Refit, Val
from megspeech.evaluation.metric.ruler import Interval
from megspeech.evaluation.scoring.score import Epoch, Score
from megspeech.model.auxiliary import Tap
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.context.timing import Clock
from megspeech.model.pyramid import PyramidConfig
from megspeech.model.words.conv import ConvConfig
from megspeech.submission.config import ReportConfig
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
    assert (balanced.val, balanced.refit) == (Val.BALANCED, Refit.VAL)


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


def test_segment_pairs_with_spanned_encoder() -> None:
    spanned = ContextConfig(encoder=Encoder.SPANNED)
    assert RunConfig(context=spanned, data=DataConfig(segment=9.0)).data.segment == 9.0
    with pytest.raises(ValueError, match="continuous segments"):
        RunConfig(context=spanned)
    with pytest.raises(ValueError, match="continuous segments"):
        RunConfig(data=DataConfig(segment=9.0))
    with pytest.raises(ValueError, match="word windows"):
        RunConfig(context=spanned, data=DataConfig(segment=9.0), augment=Augment(jitter=2))
    with pytest.raises(ValueError, match="right cut"):
        RunConfig(context=spanned, data=DataConfig(segment=9.0), augment=Augment(right_cut=0.18))


def test_saved_flat_optim_sampler(tmp_path: Path) -> None:
    """A run saved with the sampler knobs under `optim` loads them into `sampler`."""
    old = {"optim": {"epochs": 3, "priority": 0.5, "sampler": "uniform", "stratified": 2.0}}
    (tmp_path / METRICS).write_text(json.dumps({"config": old}))
    config = RunConfig.saved(tmp_path)
    assert config.optim.epochs == 3
    assert (config.sampler.priority, config.sampler.base, config.sampler.alpha) == (0.5, Base.UNIFORM, 2.0)


def test_stop_config_stalled() -> None:
    rising = [epoch(i, 0.1 * i) for i in range(6)]
    peaked = [epoch(0, 0.2), epoch(1, 0.5), epoch(2, 0.5), epoch(3, 0.4), epoch(4, 0.3)]
    assert not StopConfig().stalled(peaked)
    assert not StopConfig(3).stalled(rising)
    assert not StopConfig(4).stalled(peaked)
    assert StopConfig(3).stalled(peaked)
    assert StopConfig(1).stalled(peaked[:3])


def test_run_config_stalled() -> None:
    peaked = [epoch(0, 0.2), epoch(1, 0.4), epoch(2, 0.3), epoch(3, 0.3)]
    assert RunConfig(stop=StopConfig(2)).stalled(peaked)
    assert not RunConfig().stalled(peaked)


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


def test_report_defaults_off(tmp_path: Path) -> None:
    """A record from before the report section loads with it off."""
    (tmp_path / METRICS).write_text(json.dumps({"config": {"optim": {"epochs": 3}}}))
    assert RunConfig.saved(tmp_path).report == ReportConfig()
    assert RunConfig().report == ReportConfig(local=False, tracks=(), aligned=False)


def test_acoustic_needs_the_envelope_geometry() -> None:
    assert RunConfig(optim=OptimConfig(acoustic=0.5)).optim.acoustic == 0.5
    with pytest.raises(ValueError, match="encoder"):
        RunConfig(optim=OptimConfig(acoustic=0.5), context=ContextConfig(encoder=Encoder.CACHED))
    with pytest.raises(ValueError, match="cut"):
        RunConfig(optim=OptimConfig(acoustic=0.5), data=DataConfig(window=2.0))
    with pytest.raises(ValueError, match="acoustic head"):
        RunConfig(optim=OptimConfig(tap=Tap.TIME))


def test_listens() -> None:
    assert not OptimConfig().listens
    assert OptimConfig(acoustic=0.3).listens
    assert OptimConfig(heads=0.0, audio=1.0).listens
    with pytest.raises(ValueError, match="cut"):
        RunConfig(optim=OptimConfig(audio=1.0), data=DataConfig(window=2.0))


def test_acoustic_refuses_the_pyramid() -> None:
    with pytest.raises(ValueError, match="pyramid"):
        RunConfig(
            optim=OptimConfig(acoustic=0.5), context=ContextConfig(conv=ConvConfig(pyramid=PyramidConfig(taps=(1,))))
        )


def test_topk_mix_validated() -> None:
    assert RunConfig(optim=OptimConfig(topk=0.5, topk_temperature=2.0, topk_margin=0.0)).optim.topk == 0.5
    for optim in (OptimConfig(topk=1.5), OptimConfig(topk_temperature=0.0), OptimConfig(topk_margin=-1.0)):
        with pytest.raises(ValueError, match="optim.topk"):
            RunConfig(optim=optim)

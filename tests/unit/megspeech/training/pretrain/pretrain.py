import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.constants import CHANNELS
from megspeech.model.config import ContextConfig, Covariance
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SCALE_MEDIAN, SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.training.config import METRICS
from megspeech.training.pretrain import pretrain as module
from megspeech.training.pretrain.pretrain import (
    LOSSES,
    PREDICTOR,
    CollapseError,
    Loss,
    Predictor,
    PretrainConfig,
    Pretrainer,
    Round,
    Target,
)
from megspeech.training.pretrain.windows import Pool, Sessions

CPU = torch.device("cpu")
SMALL = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
CONFIG = PretrainConfig(context=SMALL, epochs=2, batch=4, windows=16, fit=16, monitor=2, collapse_floor=0.0, top=2)


def recording() -> DeviceRecording:
    return DeviceRecording(torch.randn(CHANNELS, 400).half(), np.array([0, 200, 400]), ("a", "b"), pd.DataFrame())


def pretrainer(config: PretrainConfig = CONFIG) -> Pretrainer:
    torch.manual_seed(0)
    return Pretrainer(config, CPU, Pool(recording(), config.data.samples))


def net(trainer: Pretrainer) -> SentenceNet:
    return trainer.net(torch.Generator().manual_seed(0))


def test_config_validates() -> None:
    with pytest.raises(ValueError, match="noise"):
        PretrainConfig(context=ContextConfig(whiten=True, covariance=Covariance.NOISE))
    with pytest.raises(ValueError, match="top"):
        PretrainConfig(context=SMALL, top=3)
    with pytest.raises(ValueError, match="ema"):
        PretrainConfig(ema=1.0)


def test_losses() -> None:
    target = torch.randn(10, 8)
    assert LOSSES[Loss.SMOOTH_L1](target, target) == 0
    assert LOSSES[Loss.COSINE](target, target).abs() < 1e-5
    assert LOSSES[Loss.COSINE](target, -target) > 1.9


def test_forward() -> None:
    predictor = Predictor(CHANNELS, 8)
    assert predictor.forward(torch.randn(2, 8, 50)).shape == (2, 50, 8)
    assert predictor.token.shape == (CHANNELS, 1)


def test_spread() -> None:
    assert Pretrainer.spread(torch.randn(512, 64))[0] > 0.9
    same = Pretrainer.spread(torch.ones(512, 64) + 1e-4 * torch.randn(512, 64))
    assert same[0] < 0.01
    assert same[1] > 0.99


def test_target_of() -> None:
    student = net(pretrainer())
    teacher = Target.of(student, 1)
    assert not any(p.requires_grad for p in teacher.net.parameters())
    assert all(a is not b for a, b in zip(student.parameters(), teacher.net.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(student.parameters(), teacher.net.parameters(), strict=True))


def test_target_follow() -> None:
    student = net(pretrainer())
    teacher = Target.of(student, 1)
    before = [p.detach().clone() for p in student.parameters()]
    with torch.no_grad():
        for p in student.parameters():
            p.add_(4.0)
    teacher.follow(student, 0.75)
    assert all(torch.allclose(t, b + 1.0) for t, b in zip(teacher.net.parameters(), before, strict=True))


def test_target_convolutional() -> None:
    assert Target.convolutional(net(pretrainer())).blocks


def test_target_read() -> None:
    trainer = pretrainer()
    x = trainer.pool.sample(3, torch.Generator().manual_seed(0))
    out = Target.of(net(trainer), 2).read(x, torch.zeros(3, dtype=torch.long))
    assert out.shape == (3, 50, 8)
    assert out.mean(dim=-1).abs().max() < 1e-3
    assert not out.requires_grad


def test_net() -> None:
    assert net(pretrainer()).state_dict()[SCALE_MEDIAN].shape == (CHANNELS, 1)


def test_decay() -> None:
    trainer = pretrainer()
    assert trainer.decay(0, 100) == CONFIG.ema_start
    assert trainer.decay(100, 100) == CONFIG.ema
    assert CONFIG.ema_start < trainer.decay(10, 100) < CONFIG.ema


def test_step() -> None:
    trainer = pretrainer()
    student = net(trainer)
    predictor = Predictor(CHANNELS, 8)
    stats = trainer.step(student, predictor, Target.of(student, 2), torch.Generator().manual_seed(0))
    assert stats.shape == (4,)
    assert torch.isfinite(stats).all()
    assert predictor.token.grad is not None
    assert next(iter(student.words.parameters())).grad is not None


def test_watch() -> None:
    trainer = pretrainer(replace(CONFIG, collapse_floor=0.1))
    trainer.watch(torch.tensor([0.3, 0.9, 0.9, 0.0]))
    with pytest.raises(CollapseError):
        trainer.watch(torch.tensor([0.3, 0.01, 0.9, 0.0]))
    with pytest.raises(CollapseError):
        trainer.watch(torch.tensor([0.3, 0.9, 0.01, 0.0]))
    with pytest.raises(FloatingPointError):
        trainer.watch(torch.tensor([float("nan"), 0.9, 0.9, 0.0]))


def test_fit() -> None:
    trainer = pretrainer()
    student = net(trainer)
    teacher, history = trainer.fit(student, Predictor(CHANNELS, 8), torch.Generator().manual_seed(0))
    assert [row.epoch for row in history] == [0, 1]
    assert all(isinstance(row, Round) and row.seconds > 0 for row in history)
    assert not all(torch.equal(a, b) for a, b in zip(student.parameters(), teacher.net.parameters(), strict=True))


def test_run() -> None:
    kept, predictor, history = pretrainer().run()
    assert isinstance(kept, SentenceNet)
    assert predictor.token.shape[0] == CHANNELS
    assert len(history) == 2
    teacher, _, _ = pretrainer(replace(CONFIG, teacher=True)).run()
    assert isinstance(teacher, SentenceNet)


def test_save(tmp_path: Path) -> None:
    trainer = pretrainer()
    kept, predictor, history = trainer.run()
    trainer.save(tmp_path / "ssl", kept, predictor, history)
    assert (tmp_path / "ssl" / PREDICTOR).is_file()
    assert len(json.loads((tmp_path / "ssl" / METRICS).read_text())["history"]) == 2
    state = torch.load(tmp_path / "ssl" / "model.pt", weights_only=True)
    fresh = SentenceNet(Scaler(torch.randn(8, CHANNELS, 50)), context=ContextConfig.load(tmp_path / "ssl"))
    fresh.adopt(state)
    assert all(torch.equal(a, b) for a, b in zip(fresh.encoder(), kept.encoder(), strict=True))
    assert torch.equal(fresh.state_dict()[SCALE_MEDIAN], kept.state_dict()[SCALE_MEDIAN])


def test_pretrainer_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[list[str]] = []
    built = recording()
    cleared = Sessions(())
    monkeypatch.setattr(module.Run, "everywhere", classmethod(lambda *_: []))
    monkeypatch.setattr(module.Sessions, "of", classmethod(lambda *_: cleared))
    monkeypatch.setattr(
        module.DeviceRecording,
        "load",
        classmethod(lambda _c, _cache, _device, names, **_: loaded.append(list(names)) or built),
    )
    monkeypatch.setattr(module.DataConfig, "run_cache", lambda *_: None)
    monkeypatch.setattr(module.DataConfig, "geometry", lambda *_, **__: None)
    out = Pretrainer.of(CONFIG, DataRoot(tmp_path), CPU)
    assert loaded == [[]]
    assert out.pool.length == CONFIG.data.samples

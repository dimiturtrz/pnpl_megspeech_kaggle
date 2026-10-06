import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.constants import CHANNELS
from megspeech.data.spans.spans import Spans
from megspeech.model.config import ContextConfig, Covariance
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SCALE_MEDIAN, SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.training.config import METRICS
from megspeech.training.pretrain import pretrain as module
from megspeech.training.pretrain.masking import Channels, Masking, Region
from megspeech.training.pretrain.objectives import LOSSES, Kind, Objective, Setup, Term
from megspeech.training.pretrain.pretrain import (
    PREDICTOR,
    CollapseError,
    Predictor,
    PretrainConfig,
    Pretrainer,
    Round,
    Student,
    Target,
)
from megspeech.training.pretrain.progress import Probing, Progress
from megspeech.training.pretrain.prototypes import Prototypes
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
    with pytest.raises(ValueError, match="distinct"):
        PretrainConfig(objectives=(Objective(Kind.IBOT), Objective(Kind.IBOT)))
    with pytest.raises(ValueError, match="distinct"):
        PretrainConfig(objectives=())
    with pytest.raises(ValueError, match="batch"):
        PretrainConfig(objectives=(Objective(Kind.KOLEO),), batch=1)
    assert PretrainConfig().objectives == (Objective(Kind.REGRESSION, 1.0),)


def test_forward() -> None:
    predictor = Predictor(CHANNELS, 8)
    assert predictor.forward(torch.randn(2, 8, 50)).shape == (2, 50, 8)
    assert predictor.token.shape == (CHANNELS, 1)


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


def test_target_maps() -> None:
    trainer = pretrainer()
    teacher = Target.of(net(trainer), 2)
    x = trainer.pool.sample(3, torch.Generator().manual_seed(0))
    maps = teacher.maps(teacher.prepare(x, torch.zeros(3, dtype=torch.long)))
    assert len(maps) == 2
    assert not maps[-1].requires_grad


def test_target_prepare() -> None:
    trainer = pretrainer()
    student = net(trainer)
    x = trainer.pool.sample(3, torch.Generator().manual_seed(0))
    subject = torch.zeros(3, dtype=torch.long)
    assert torch.equal(Target.of(student, 2).prepare(x, subject), student.prepare(x, subject))


def test_target_target() -> None:
    trainer = pretrainer()
    teacher = Target.of(net(trainer), 2)
    x = trainer.pool.sample(3, torch.Generator().manual_seed(0))
    out = teacher.target(teacher.maps(teacher.prepare(x, torch.zeros(3, dtype=torch.long))))
    assert out.shape[0] == 3
    assert out.mean(dim=-1).abs().max() < 1e-3


def test_net() -> None:
    assert net(pretrainer()).state_dict()[SCALE_MEDIAN].shape == (CHANNELS, 1)


def test_decay() -> None:
    trainer = pretrainer()
    assert trainer.decay(0, 100) == CONFIG.ema_start
    assert trainer.decay(100, 100) == CONFIG.ema
    assert CONFIG.ema_start < trainer.decay(10, 100) < CONFIG.ema


ALL = (Objective(Kind.REGRESSION), Objective(Kind.IBOT, 0.5), Objective(Kind.DINO, 0.5), Objective(Kind.KOLEO, 0.1))
PROTOS = Prototypes(k=8, bottleneck=6, hidden=10, perplexity_floor=1.0)


def terms_of(trainer: Pretrainer, predictor: Predictor) -> list[Term]:
    config = trainer.config
    setup = Setup(predictor, config.context.conv.hidden, config.prototypes, config.loss, config.collapse_floor, CPU)
    return [objective.build(setup) for objective in config.objectives]


def test_show() -> None:
    trainer = pretrainer()
    student = net(trainer)
    predictor = Predictor(CHANNELS, 8)
    x = trainer.pool.sample(4, torch.Generator().manual_seed(0))
    subject = torch.zeros(4, dtype=torch.long)
    shown, hidden = trainer.show(Student(student, predictor), x, subject, torch.Generator().manual_seed(1))
    assert hidden.shape[0] == 4
    assert hidden.any()
    assert (shown.transpose(1, 2)[hidden] == predictor.token.detach().T).all()


def test_show_region() -> None:
    config = replace(CONFIG, masking=Masking(channels=Channels.REGION, region=4))
    positions = torch.randn(CHANNELS, 3)
    trainer = Pretrainer(config, CPU, Pool(recording(), config.data.samples), Region.of(positions, 4))
    student = net(trainer)
    seen: list[torch.Tensor] = []
    prepare = student.prepare
    student.prepare = lambda x, subject: seen.append(x.clone()) or prepare(x, subject)
    x = trainer.pool.sample(4, torch.Generator().manual_seed(0))
    trainer.show(
        Student(student, Predictor(CHANNELS, 8)), x, torch.zeros(4, dtype=torch.long), torch.Generator().manual_seed(1)
    )
    silenced = seen[0].eq(0).all(dim=2)
    assert (silenced.sum(dim=1) == 5).all()
    assert torch.equal(seen[0][~silenced], x[~silenced])


def test_sensors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert Pretrainer.sensors(CONFIG, DataRoot(tmp_path), CPU).nearest.numel() == 0
    positions = torch.randn(CHANNELS, 3)
    home = type("Home", (), {"home": staticmethod(lambda _root: tmp_path)})
    monkeypatch.setattr(module, "NEUROMAG", home)
    monkeypatch.setattr(
        module.Neuromag, "of", classmethod(lambda *_: type("N", (), {"positions": positions.numpy()})())
    )
    config = replace(CONFIG, masking=Masking(channels=Channels.REGION, region=6))
    assert Pretrainer.sensors(config, DataRoot(tmp_path), CPU).nearest.shape == (CHANNELS, 7)


def test_region_needs_sensors() -> None:
    config = replace(CONFIG, masking=Masking(channels=Channels.REGION))
    with pytest.raises(ValueError, match="positions"):
        Pretrainer(config, CPU, Pool(recording(), config.data.samples))


def test_step() -> None:
    trainer = pretrainer()
    student = net(trainer)
    predictor = Predictor(CHANNELS, 8)
    stats = trainer.step(
        Student(student, predictor),
        Target.of(student, 2),
        terms_of(trainer, predictor),
        torch.Generator().manual_seed(0),
    )
    assert set(stats) == {"loss", "loss/regression", "spread", "target_spread", "target_centre"}
    assert all(torch.isfinite(value) for value in stats.values())
    assert predictor.token.grad is not None


def test_step_default_matches_data2vec() -> None:
    trainer = pretrainer()
    student = net(trainer)
    predictor = Predictor(CHANNELS, 8)
    teacher = Target.of(student, 2)
    twin, twin_predictor, twin_teacher = copy.deepcopy(student), copy.deepcopy(predictor), copy.deepcopy(teacher)
    torch.manual_seed(5)
    stats = trainer.step(
        Student(student, predictor), teacher, terms_of(trainer, predictor), torch.Generator().manual_seed(7)
    )
    torch.manual_seed(5)
    generator = torch.Generator().manual_seed(7)
    config = trainer.config
    x = trainer.pool.sample(config.batch, generator)
    subject = torch.zeros(config.batch, dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        clean = twin_teacher.maps(twin_teacher.prepare(x, subject))
        target = twin_teacher.target(clean)
        shown, hidden = trainer.show(Student(twin, twin_predictor), x, subject, generator)
        pred = twin_predictor(twin_teacher.convolutional(twin).taps(shown)[-1])
        legacy = LOSSES[config.loss](pred[hidden].float(), target[hidden])
    assert torch.equal(stats["loss"], legacy)
    assert torch.equal(stats["loss/regression"], legacy)


def test_step_all_objectives() -> None:
    config = replace(CONFIG, objectives=ALL, prototypes=PROTOS)
    trainer = pretrainer(config)
    student = net(trainer)
    predictor = Predictor(CHANNELS, 8)
    terms = terms_of(trainer, predictor)
    stats = trainer.step(
        Student(student, predictor), Target.of(student, 2), terms, torch.Generator().manual_seed(0), 0.5
    )
    assert {"loss/ibot", "loss/dino", "loss/koleo", "ibot/perplexity", "dino/entropy", "koleo/nearest"} <= set(stats)
    expected = torch.stack([term.weight * stats[f"loss/{term.name}"] for term in terms]).sum()
    assert torch.allclose(stats["loss"], expected)
    assert all(p.grad is not None for term in terms for p in term.parameters())


def test_clear() -> None:
    trainer = pretrainer(replace(CONFIG, objectives=ALL, prototypes=PROTOS))
    student = net(trainer)
    predictor = Predictor(CHANNELS, 8)
    terms = terms_of(trainer, predictor)
    for p in [*student.parameters(), *predictor.parameters(), *(p for t in terms for p in t.parameters())]:
        p.grad = torch.ones_like(p)
    Pretrainer.clear(Student(student, predictor), terms)
    assert all(p.grad is None for p in [*student.parameters(), *predictor.parameters()])
    assert all(p.grad is None for term in terms for p in term.parameters())


def test_average() -> None:
    stats = [{"loss": torch.tensor(1.0), "a": torch.tensor(2.0)}, {"loss": torch.tensor(3.0), "a": torch.tensor(4.0)}]
    assert Pretrainer.average(stats) == {"loss": 2.0, "a": 3.0}


def test_watch() -> None:
    floors = {"spread": 0.1, "ibot/perplexity": 2.0}
    healthy = {"loss": 0.3, "spread": 0.9, "ibot/perplexity": 50.0}
    Pretrainer.watch(healthy, floors)
    with pytest.raises(CollapseError, match="spread"):
        Pretrainer.watch(healthy | {"spread": 0.01}, floors)
    with pytest.raises(CollapseError, match="perplexity"):
        Pretrainer.watch(healthy | {"ibot/perplexity": 1.1}, floors)
    with pytest.raises(FloatingPointError):
        Pretrainer.watch(healthy | {"loss": float("nan")}, floors)
    Pretrainer.watch({"loss": 1.0}, {})


def test_fit() -> None:
    trainer = pretrainer()
    student = net(trainer)
    teacher, history = trainer.fit(student, Predictor(CHANNELS, 8), torch.Generator().manual_seed(0))
    assert [row.epoch for row in history] == [0, 1]
    assert all(isinstance(row, Round) and row.seconds > 0 for row in history)
    assert not all(torch.equal(a, b) for a, b in zip(student.parameters(), teacher.net.parameters(), strict=True))


def test_fit_all_objectives() -> None:
    config = replace(CONFIG, objectives=ALL, prototypes=PROTOS, monitor=1)
    trainer = pretrainer(config)
    _, history = trainer.fit(net(trainer), Predictor(CHANNELS, 8), torch.Generator().manual_seed(0))
    assert all(math.isfinite(v) for row in history for v in row.stats.values())
    assert {"loss/ibot", "ibot/perplexity", "dino/entropy"} <= set(history[-1].stats)
    collapsing = replace(config, prototypes=replace(PROTOS, perplexity_floor=99.0))
    with pytest.raises(CollapseError, match="perplexity"):
        pretrainer(collapsing).fit(net(trainer), Predictor(CHANNELS, 8), torch.Generator().manual_seed(0))


class Fixed(Progress):
    """A probe reading a fixed score in place of embedding anything."""

    @override
    def read(self, student: SentenceNet, teacher: SentenceNet) -> dict[str, float]:
        return {"probe/student": 0.25, "probe/teacher": 0.5}


def fixed() -> Fixed:
    labelled = torch.arange(4)
    split = Split.from_arrays(
        np.zeros((4, 2, 2), dtype=np.float32), np.arange(4), Spans.by_gap(np.arange(4) * 9.0), CPU
    )
    return Fixed(split, split, labelled, labelled, CPU)


def test_probed() -> None:
    config = replace(CONFIG, epochs=3, probe=Probing(every=2))
    trainer = Pretrainer(config, CPU, Pool(recording(), config.data.samples), progress=fixed())
    teacher = Target.of(net(trainer), 2)
    assert [bool(trainer.probed(teacher.net, teacher, epoch)) for epoch in range(3)] == [False, True, True]
    assert trainer.probed(teacher.net, teacher, 1) == {"probe/student": 0.25, "probe/teacher": 0.5}
    off = Pretrainer(replace(config, probe=Probing(every=0)), CPU, trainer.pool, progress=fixed())
    assert off.probed(teacher.net, teacher, 1) == {}
    assert pretrainer(config).probed(teacher.net, teacher, 1) == {}


def test_fit_probe() -> None:
    trainer = Pretrainer(CONFIG, CPU, Pool(recording(), CONFIG.data.samples), progress=fixed())
    _, history = trainer.fit(net(trainer), Predictor(CHANNELS, 8), torch.Generator().manual_seed(0))
    assert [row.epoch for row in history] == [-1, 0, 1]
    assert history[0].probe == {"probe/student": 0.25, "probe/teacher": 0.5}
    assert not history[1].probe
    assert history[2].probe["probe/teacher"] == 0.5


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
    monkeypatch.setattr(module.Progress, "load", classmethod(lambda *_: fixed()))
    out = Pretrainer.of(CONFIG, DataRoot(tmp_path), CPU)
    assert loaded == [[]]
    assert out.pool.length == CONFIG.data.samples
    assert isinstance(out.progress, Fixed)
    assert Pretrainer.of(replace(CONFIG, probe=Probing(every=0)), DataRoot(tmp_path), CPU).progress is None

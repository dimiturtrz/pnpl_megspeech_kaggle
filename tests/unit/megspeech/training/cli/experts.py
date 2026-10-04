import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.expert import Crop
from megspeech.training.cli import experts as module
from megspeech.training.cli.experts import MODEL, Assembly, Run, main
from megspeech.training.config import METRICS, RunConfig

SMALL = ConvConfig(hidden=4, dilations=(1,))
SHAPE = {"width": 16, "timescales": 4, "layers": 1, "heads": 2}
WINDOWS = ((1.0, 0.0), (0.2, 0.0), (0.2, 0.2))
GENERATOR = torch.Generator().manual_seed(0)


def expert(window: float, delay: float) -> Run:
    config = RunConfig(data=DataConfig(window=window, delay=delay), context=ContextConfig(conv=SMALL, **SHAPE))
    sample = torch.randn(8, CHANNELS, 50, generator=GENERATOR) * 3 + 1
    return Run(config, SentenceNet(Scaler(sample), context=config.context).state_dict())


def stack() -> Run:
    context = ContextConfig(conv=replace(SMALL, segments=len(WINDOWS)), encoder=Encoder.TRAINED, **SHAPE)
    config = RunConfig(data=DataConfig(features="a+b+c"), context=context)
    width = context.conv.width
    sample = torch.randn(8, width, 1, generator=GENERATOR) * 2 + 0.5
    return Run(config, SentenceNet(Scaler(sample), context=context, channels=width).state_dict())


def assembly() -> Assembly:
    return Assembly(stack(), tuple(expert(window, delay) for window, delay in WINDOWS))


def loaded(run: Run) -> SentenceNet:
    channels = run.state["scale.median"].shape[0]
    net = SentenceNet(Scaler(torch.zeros(1, channels, 1)), context=run.config.context, channels=channels)
    net.load_state_dict(run.state)
    return net.eval()


def test_load(tmp_path: Path) -> None:
    run = expert(0.2, 0.0)
    torch.save(run.state, tmp_path / MODEL)
    (tmp_path / METRICS).write_text(json.dumps({"config": asdict(run.config)}))
    again = Run.load(tmp_path)
    assert again.config == run.config
    assert torch.equal(again.state["scale.median"], run.state["scale.median"])


def test_plain() -> None:
    run = expert(0.2, 0.0)
    assert run.plain()
    assert not stack().plain()
    assert not replace(run, config=replace(run.config, data=DataConfig(baseline=0.1))).plain()


def test_weights() -> None:
    state = expert(0.2, 0.0).state
    kept = Run(RunConfig(), state).weights(("scale.",), kept=True)
    dropped = Run(RunConfig(), state).weights(("scale.",), kept=False)
    assert set(kept) == {"scale.median", "scale.spread"}
    assert set(kept) | set(dropped) == set(state)
    assert not set(kept) & set(dropped)


def test_config() -> None:
    config = assembly().config(rhythm=True)
    assert config.context.encoder is Encoder.EXPERTS
    assert config.context.crops == (Crop(0, 50), Crop(0, 10), Crop(10, 10))
    assert config.context.conv == SMALL
    assert config.context.rhythm
    assert config.context.width == SHAPE["width"]
    assert (config.data.window, config.data.delay, config.data.features) == (1.0, 0.0, "")
    mixed = Assembly(stack(), (expert(1.0, 0.0), replace(expert(0.2, 0.0), config=RunConfig())))
    with pytest.raises(ValueError, match="one CNN shape"):
        mixed.config()


def test_offset() -> None:
    assert Assembly.offset(DataConfig(delay=0.2)) == 10
    assert Assembly.offset(DataConfig(delay=0.2, lead=0.1)) == 5


def test_end() -> None:
    assert Assembly.end(DataConfig(window=0.2, delay=0.2)) == 20
    assert Assembly.end(DataConfig(window=1.0, lead=0.1)) == 50


def test_crop() -> None:
    window = DataConfig(window=1.0)
    assert Assembly.crop(window, DataConfig(window=0.2, delay=0.2)) == (10, 10)
    with pytest.raises(ValueError, match="outside the shared window"):
        Assembly.crop(window, DataConfig(window=0.2, delay=0.9))
    with pytest.raises(ValueError, match="outside the shared window"):
        Assembly.crop(window, DataConfig(window=0.2, lead=0.1))


def test_placement() -> None:
    placed = Assembly.placement(DataConfig(features="x", subjects=(3,)), DataConfig(window=0.5, delay=0.2, decimate=4))
    assert placed == DataConfig(window=0.5, delay=0.2, decimate=4, subjects=(3,))


def test_state() -> None:
    parts = assembly()
    config = parts.config()
    composite = loaded(Run(config, parts.state(config)))
    x = torch.randn(4, CHANNELS, 50, generator=GENERATOR) * 4 + 1
    x[0, :, 30:] = 0
    subject = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        pairs = zip(parts.experts, config.context.crops, strict=True)
        joined = torch.cat([loaded(run).embed(crop(x), subject) for run, crop in pairs], dim=-1)
        expected = loaded(parts.stack).embed(joined[..., None], subject)
        assert torch.allclose(composite.embed(x, subject), expected, atol=1e-4)
    broken = replace(parts, stack=replace(parts.stack, state=parts.stack.state | {"stray": torch.zeros(1)}))
    with pytest.raises(ValueError, match="no place"):
        broken.state(config)


def test_save(tmp_path: Path) -> None:
    assembly().save(tmp_path / "joint", rhythm=True)
    net = SentenceNet.load(tmp_path / "joint" / "model.pt")
    assert any(name.startswith("timing.") for name in net.state_dict())
    assert RunConfig.saved(tmp_path / "joint").context.encoder is Encoder.EXPERTS


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = assembly()
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    runs = {"s": parts.stack} | {f"e{index}": run for index, run in enumerate(parts.experts)}
    monkeypatch.setattr(module.Run, "load", classmethod(lambda _, path: runs[path.name]))
    monkeypatch.setattr(sys, "argv", ["experts", "--stack", "s", "--experts", "e0", "e1", "e2", "--name", "joint"])
    main()
    assert (DataRoot(tmp_path).models() / "joint" / "model.pt").exists()

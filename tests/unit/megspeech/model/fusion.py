from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.model.branch import Branch
from megspeech.model.checkpoint import Checkpoint
from megspeech.model.config import ContextConfig
from megspeech.model.fusion import PRIMARY, Fusion
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)
SLOT = torch.tensor([[0, 1, 3], [2, 3, 3]])
ONSET = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
CPU = torch.device("cpu")


def saved(directory: Path) -> Path:
    directory.mkdir(parents=True)
    scale = Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR))
    torch.save(SentenceNet(scale, CONTEXT).state_dict(), directory / "model.pt")
    CONTEXT.save(directory)
    return directory


def branch(directory: Path) -> Branch:
    return Branch.load(saved(directory), CONTEXT, CHANNELS)


def inputs() -> SpanInput:
    return SpanInput(
        torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR), ONSET, SLOT, torch.zeros(3, dtype=torch.long)
    )


def test_names(tmp_path: Path) -> None:
    assert Fusion([branch(tmp_path / "stem")], 0.3).names() == (PRIMARY, "stem")


def test_gates(tmp_path: Path) -> None:
    fusion = Fusion([branch(tmp_path / "a"), branch(tmp_path / "b")], 0.5)
    assert torch.equal(fusion.eval().gates(4, CPU), torch.ones(4, 3))
    dropped = fusion.train().gates(2000, CPU)
    assert (dropped > 0).any(dim=1).all()
    assert torch.allclose(dropped.sum(1), torch.full((2000,), 3.0))
    assert (dropped == 0).any()
    assert abs(dropped.mean().item() - 1.0) < 0.05
    assert torch.equal(Fusion([], 0.5).train().gates(5, CPU), torch.ones(5, 1))


def test_mix(tmp_path: Path) -> None:
    fusion = Fusion([branch(tmp_path / "a")], 0.3).eval()
    parts = [torch.randn(2, 3, 4, generator=GENERATOR), torch.randn(2, 3, 4, generator=GENERATOR)]
    assert torch.allclose(fusion.mix(parts), parts[0] + parts[1])
    with fusion.silenced(PRIMARY):
        assert torch.allclose(fusion.mix(parts), parts[1])


def test_silenced(tmp_path: Path) -> None:
    fusion = Fusion([branch(tmp_path / "a")], 0.3).eval()
    with fusion.silenced("a"):
        assert fusion.gates(1, CPU).tolist() == [[1.0, 0.0]]
    assert fusion.gates(1, CPU).tolist() == [[1.0, 1.0]]
    with pytest.raises(KeyError, match="no source"), fusion.silenced("missing"):
        pass


def test_encoders(tmp_path: Path) -> None:
    made = branch(tmp_path / "a")
    (params,) = Fusion([made], 0.3).encoders()
    assert {id(p) for p in params} == {id(p) for p in made.encoder()}
    assert not {id(p) for p in params} & {id(p) for p in made.project.parameters()}


def test_fused_net_equals_solo_when_the_branch_adds_nothing(tmp_path: Path) -> None:
    sample = torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)
    solo = SentenceNet(Scaler(sample), CONTEXT).eval()
    fused = (
        SentenceNet(Scaler(sample), replace(CONTEXT, fused=("a",))).fuse(Fusion([branch(tmp_path / "a")], 0.3)).eval()
    )
    fused.load_state_dict(solo.state_dict(), strict=False)
    nn.init.zeros_(fused.fusion.branches[0].project.weight)
    nn.init.zeros_(fused.fusion.branches[0].project.bias)
    args = inputs()
    for a, b in zip(solo(args), fused(args), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_fused_net_reloads_and_silences(tmp_path: Path) -> None:
    models = tmp_path / "models"
    saved(models / "a")
    context = replace(CONTEXT, fused=("a",))
    branches = [Branch.load(models / "a", context, CHANNELS)]
    net = SentenceNet(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), context).fuse(
        Fusion(branches, 0.3)
    )
    (models / "run").mkdir()
    torch.save(net.state_dict(), models / "run" / "model.pt")
    context.save(models / "run")
    loaded = Checkpoint.load(models / "run" / "model.pt").eval()
    args = inputs()
    assert torch.allclose(net.eval()(args)[0], loaded(args)[0], atol=1e-6)
    with loaded.fusion.silenced("a"):
        assert not torch.allclose(net(args)[0], loaded(args)[0], atol=1e-6)

from pathlib import Path
from types import SimpleNamespace
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.model.megxl import CHECKPOINT, LATENT, MAGNETOMETER, ROTATIONS, Checkpoint, Geometry, Trunk

GRADIOMETER = 3012


def channel(name: str, coil: int, position: float) -> dict[str, object]:
    loc = np.zeros(12)
    loc[:3] = position
    loc[3:6] = (1.0, 0.0, 0.0)
    loc[9:12] = (0.0, 0.0, 1.0)
    return {"ch_name": name, "coil_type": coil, "loc": loc}


CHANNELS = [
    channel("MEG 0112", GRADIOMETER, 1.0),
    channel("EEG 001", MAGNETOMETER, 9.0),
    channel("MEG 0111", MAGNETOMETER, -1.0),
]


class Features(nn.Module):
    """The trunk's interface: features of `raw` one token per 2 samples, a sum of the sensor inputs."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    @override
    def forward(self, raw: Tensor, xyz: Tensor, abc: Tensor, kind: Tensor, *, apply_mask: bool) -> dict[str, Tensor]:
        assert not apply_mask
        tokens = raw.unfold(-1, 2, 2).mean(-1, keepdim=True)
        sensors = (xyz.sum(-1) + abc.sum(-1) + kind)[..., None, None]
        return {"features": (self.scale * tokens + sensors).expand(-1, -1, -1, LATENT)}

    def enable_gradient_checkpointing(self) -> None:
        self.checkpointed = True


def test_of():
    geometry = Geometry.of(CHANNELS)
    assert geometry.kind.tolist() == [1, 0]
    np.testing.assert_allclose(geometry.xyz, [[-1 / 3] * 3, [1 / 3] * 3])
    np.testing.assert_allclose(geometry.abc, [[0, 0, 1], [1, 0, 0]])
    assert np.sqrt(3 * np.mean(np.sum(geometry.xyz**2, axis=1))) == pytest.approx(1.0)


def test_neuromag(monkeypatch: pytest.MonkeyPatch):
    read = SimpleNamespace(read_info=lambda path, verbose: {"chs": CHANNELS, "path": path, "verbose": verbose})
    monkeypatch.setattr("importlib.import_module", lambda name: read if name == "mne.io" else None)
    assert Geometry.neuromag(Path("x.fif")).kind.tolist() == [1, 0]


def test_under(tmp_path: Path):
    root = DataRoot(tmp_path)
    assert Checkpoint.under(root).weights == root.foreign("MEG-XL") / CHECKPOINT


def test_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.path", [])
    imported: list[str] = []
    monkeypatch.setattr("importlib.import_module", imported.append)
    Checkpoint(tmp_path / "w.ckpt", tmp_path).modules()
    assert imported == ["brainstorm.neuro_tokenizers.biocodec.model", "brainstorm.models.criss_cross_transformer"]


def saved(path: Path, state: dict[str, Tensor]) -> Checkpoint:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hyper_parameters": {"scale": 1.0, "debug_nan_checks": False}, "state_dict": state}, path)
    return Checkpoint(path, path.parent)


def faked(monkeypatch: pytest.MonkeyPatch) -> None:
    biocodec = SimpleNamespace(BioCodecModel=SimpleNamespace(_get_optimized_model=lambda: "tokenizer"))
    crisscross = SimpleNamespace(CrissCrossTransformerModule=lambda tokenizer, scale: Features(scale))
    monkeypatch.setattr(Checkpoint, "modules", lambda _self: (biocodec, crisscross))


def test_trunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    faked(monkeypatch)
    path = tmp_path / "w.ckpt"
    module = saved(path, {"scale": torch.tensor(3.0), f"layer.{ROTATIONS}": torch.zeros(2)}).trunk()
    assert module.state_dict()["scale"].item() == 3.0
    assert module.checkpointed
    with pytest.raises(ValueError, match="unexpected"):
        saved(path, {"scale": torch.tensor(3.0), "other": torch.zeros(1)}).trunk()


def test_forward():
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    x = torch.ones(3, 2, 8)
    features = trunk.forward(x)
    assert features.shape == (3, 2, 4, LATENT)
    expected = torch.tensor([-1 + 1 + 1, 1 + 1 + 0], dtype=torch.float32) + 1
    torch.testing.assert_close(features[0, :, 0, 0], expected)


def test_chunk(monkeypatch: pytest.MonkeyPatch):
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    x = torch.arange(80.0).reshape(5, 2, 8)
    whole = trunk.chunk(x)
    calls: list[int] = []
    chunk = trunk.chunk
    monkeypatch.setattr("megspeech.model.megxl.ATTENTION_BATCH", 4)
    monkeypatch.setattr(trunk, "chunk", lambda part: calls.append(len(part)) or chunk(part))
    torch.testing.assert_close(trunk.forward(x), whole)
    assert calls == [2, 2, 1]


def test_pretrained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    faked(monkeypatch)
    root = DataRoot(tmp_path)
    saved(Checkpoint.under(root).weights, {"scale": torch.tensor(2.0)})
    monkeypatch.setattr(Geometry, "neuromag", classmethod(lambda cls, _path: cls.of(CHANNELS)))
    trunk = Trunk.pretrained(root)
    assert trunk.kind.tolist() == [1, 0]


def test_groups():
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    model = nn.Sequential(nn.Linear(2, 2), trunk)
    groups = Trunk.groups(model, [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=1e-4)
    ids = [{id(p) for p in group.pop("params")} for group in groups]
    assert ids == [{id(p) for p in model[0].parameters()}, {id(p) for p in trunk.parameters()}]
    assert groups == [{"weight_decay": 0.0}, {"lr": 1e-4}]
    assert all(p.requires_grad for p in trunk.parameters())
    Trunk.groups(model, [{"params": list(model.parameters())}], lr=0.0)
    assert not any(p.requires_grad for p in trunk.parameters())

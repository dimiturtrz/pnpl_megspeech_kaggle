from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast, override

import pytest
import torch
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Partition
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.studies.experts import gates as module
from megspeech.evaluation.studies.experts.gates import COMPONENTS, FOLDS, TAPS, Expert, Features, Gate, Head
from megspeech.evaluation.studies.experts.peer import Study
from megspeech.evaluation.studies.pos.syntax import Embedded
from megspeech.model.auxiliary import Tap
from tests.unit.megspeech.evaluation.studies.experts.boosted import VOCAB, rows

COUNT = 60
DIM = 8


@dataclass(frozen=True)
class Scored(Study):
    @override
    def official(self, logp: Tensor) -> float:
        return float(logp.max(-1).values.mean())


class Reading:
    def __call__(self, _: Split) -> dict[Tap, object]:
        return {Tap.WORD: torch.eye(DIM).numpy(), Tap.CONTEXT: torch.zeros(DIM, DIM).numpy()}


def grouped() -> object:
    base = rows(COUNT)
    group = torch.arange(COUNT) % FOLDS
    return type(base)(base.base, base.tags, base.lexical, base.gaps, base.y, base.near, group)


def gate(names: tuple[str, ...] = ("textword", "audsent")) -> Gate:
    data = grouped()
    study = Scored(data, data, ("a",), cast("Evaluator", None), torch.zeros(VOCAB), {}, torch.device("cpu"))  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(1)
    return Gate(
        study, {name: Features(torch.randn(COUNT, DIM, generator=generator), torch.randn(COUNT, DIM)) for name in names}
    )


def test_tap() -> None:
    assert Expert("textsent").tap is TAPS["textsent"] is Tap.CONTEXT
    assert Expert("audword").tap is Tap.WORD


def test_model() -> None:
    assert Expert("audword").model(1) == "xp-audword-1"
    assert Expert("audword").model(None) == "xp-audword-full"


def test_vectors() -> None:
    weight = torch.eye(DIM)
    head = Head(cast("Embedded", Reading()), Tap.WORD, weight * 3, torch.zeros(DIM))
    out = head.vectors(cast("Split", None))
    assert out.shape == (DIM, DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(DIM))


def test_reduced() -> None:
    features = Features(torch.randn(COUNT, DIM), torch.randn(COUNT // 2, DIM)).reduced(3)
    assert features.oof.shape == (COUNT, 3)
    assert features.val.shape == (COUNT // 2, 3)
    assert COMPONENTS > 0


def test_design() -> None:
    chosen = gate()
    data = chosen.study.oof
    assert chosen.design(data, (), val=False).x.shape == (COUNT, VOCAB)
    assert chosen.design(data, ("textword",), val=False).x.shape == (COUNT, VOCAB + 3 * DIM + 2)
    assert chosen.design(data, ("textword", "audsent"), val=True).x.shape == (COUNT, VOCAB + 6 * DIM + 2)


def test_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EPOCHS", 2)
    out = gate().arm("x", ("textword",))
    assert out["arm"] == "x"
    assert len(cast("list[float]", out["official"])) == len(module.SEEDS)


def test_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EPOCHS", 1)
    labels = [row["arm"] for row in gate().arms()]
    assert labels == ["control", "+textword", "+audsent", "all", "all-textword", "all-audsent"]


def test_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EPOCHS", 2)
    epochs, models = gate().trained(("textword",))
    assert 1 <= epochs <= 2
    assert len(models) == len(module.SEEDS)


def test_averaged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EPOCHS", 1)
    chosen = gate()
    _, models = chosen.trained(("textword",))
    out = chosen.averaged(models, ("textword",))
    assert out.shape == (COUNT, VOCAB)
    assert torch.allclose(out.sum(-1), torch.ones(COUNT), atol=1e-4)


def test_submissions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "EPOCHS", 1)
    out = gate().submissions({"control": (), "textword": ("textword",)}, gate())
    assert set(out) == {"control", "textword"}
    assert out["textword"][1].shape == (COUNT, VOCAB)


def test_full(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    head = Head(cast("Embedded", Reading()), Tap.WORD, torch.eye(DIM), torch.zeros(DIM))
    monkeypatch.setattr(module.Head, "load", staticmethod(lambda *_: head))
    root = SimpleNamespace(models=lambda: tmp_path)
    out = Features.full(Expert("textword"), cast("DataRoot", root), [cast("Split", None)] * 2, torch.device("cpu"))
    assert out.shape == (2 * DIM, DIM)


def test_residual() -> None:
    out = gate().residual("textword")
    assert set(out) == {"error", "margin"}


def test_held_out_r2() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(90, 4, generator=generator)
    group = torch.arange(90) % FOLDS
    assert Gate.held_out_r2(x, x[:, 0] * 2, group) > 0.9
    assert Gate.held_out_r2(x, torch.randn(90, generator=generator), group) < 0.1


def test_held(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each requested fold's held-out runs are the validation runs its model's data config leaves out."""
    runs = [SimpleNamespace(name=name) for name in "abc"]
    split = SimpleNamespace(split=lambda everyone: {Partition.VALIDATION: everyone[:1]})
    monkeypatch.setattr(module.Run, "everywhere", staticmethod(lambda _: runs))
    monkeypatch.setattr(module.DataConfig, "shared", staticmethod(lambda *_: split))
    root = SimpleNamespace(models=lambda: tmp_path)
    assert Expert("textword").held(cast("DataRoot", root), [0, 2]) == [{"a"}, {"a"}]

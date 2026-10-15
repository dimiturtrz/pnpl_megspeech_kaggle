import json
from pathlib import Path
from typing import override

import torch
from torch import nn

from megspeech.data.batching.split import SpanInput
from megspeech.data.config import METRICS
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.encoder.branch import SCALE_MEDIAN
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import SubjectLayer, Unseen
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.megxl import POOLED

SAMPLES = 50


class Peaked(nn.Module):
    def __init__(self, word: int) -> None:
        super().__init__()
        self.word = word

    @override
    def forward(self, spans: SpanInput) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(*spans.slot.shape, VOCAB_SIZE)
        logits[..., self.word] = 10
        return logits, logits


def test_forward() -> None:
    slot = torch.tensor([[0, 1]])
    members: list[nn.Module] = [Peaked(3), Peaked(3), Peaked(7)]
    context, word = Ensemble(members).forward(
        SpanInput(torch.zeros(2, 1, 1), torch.zeros(1, 2), slot, torch.zeros(2, dtype=torch.long))
    )
    assert torch.all(context.argmax(-1) == 3)
    assert torch.allclose(context, word)


def test_load(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    model = SentenceNet(Scaler(torch.randn(8, CHANNELS, SAMPLES, generator=generator)))
    torch.save(model.state_dict(), tmp_path / "model.pt")
    loaded = Ensemble.load([tmp_path / "model.pt"] * 2)
    assert len(loaded.members) == 2
    restored = loaded.members[0].state_dict()
    assert all(torch.equal(restored[key], value) for key, value in model.state_dict().items())


def test_load_shaped(tmp_path: Path) -> None:
    context = ContextConfig(width=32, heads=2, layers=1)
    model = SentenceNet(Scaler(torch.zeros(1, CHANNELS, 1)), context=context)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    context.save(tmp_path)
    assert Ensemble.load([tmp_path / "model.pt"]).members[0].state_dict()["head.weight"].shape[1] == context.width


def test_load_unseen(tmp_path: Path) -> None:
    context = ContextConfig(width=32, heads=2, layers=1, subjects=3)
    model = SentenceNet(Scaler(torch.zeros(1, CHANNELS, 1)), context=context)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    context.save(tmp_path)
    layer = Ensemble.load([tmp_path / "model.pt"], Unseen.MEAN).members[0].subjects
    assert isinstance(layer, SubjectLayer)
    assert layer.unseen is Unseen.MEAN


def test_load_cached(tmp_path: Path) -> None:
    context = ContextConfig(width=32, heads=2, layers=1, encoder=Encoder.CACHED)
    model = SentenceNet(Scaler(torch.zeros(1, POOLED, 1)), context=context)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    context.save(tmp_path)
    loaded = Ensemble.load([tmp_path / "model.pt"]).members[0].state_dict()
    assert loaded[SCALE_MEDIAN].shape == (POOLED, 1)


def test_trained(tmp_path: Path) -> None:
    assert Ensemble.trained(tmp_path, "run") == [tmp_path / "run"]
    (tmp_path / "trained").mkdir()
    (tmp_path / "trained" / METRICS).write_text(json.dumps({"best": {}}))
    assert Ensemble.trained(tmp_path, "trained") == [tmp_path / "trained"]
    (tmp_path / "ens").mkdir()
    (tmp_path / "ens" / METRICS).write_text(json.dumps({"members": ["a", "b"]}))
    assert Ensemble.trained(tmp_path, "ens") == [tmp_path / "a", tmp_path / "b"]

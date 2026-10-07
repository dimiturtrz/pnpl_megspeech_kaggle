from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.spanned import WORD, SpannedConfig, SpanWords
from megspeech.model.words.trunk import DOWNSAMPLING, Geometry, Trunk
from tests.unit.megspeech.model.words.trunk import CHANNELS, Features

SENSORS = 2
torch.manual_seed(0)
TINY = SpannedConfig(components=3, width=5)


def words(spanned: SpannedConfig = TINY) -> SpanWords:
    return SpanWords(Trunk(Features(), Geometry.of(CHANNELS)), ConvConfig(), spanned, SENSORS)


def test_pretrained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    monkeypatch.setattr(Trunk, "pretrained", classmethod(lambda _cls, _root: trunk))
    assert SpanWords.pretrained(DataRoot(tmp_path), ConvConfig(), TINY, SENSORS).trunk is trunk


def test_backbone():
    encoder = words()
    backbone = encoder.backbone()
    assert {id(p) for p in backbone} == {id(p) for p in encoder.trunk.parameters()}
    assert not {id(p) for p in backbone} & {id(p) for p in encoder.mix.parameters()}


def test_positions():
    encoder = words()
    x = torch.randn(2, SENSORS, 3 * DOWNSAMPLING)
    onset = torch.tensor([[0.0, 0.3], [0.0, 0.0]])
    slot = torch.tensor([[0, 0], [1, 2]])
    out = encoder.positions(x, onset, slot)
    assert out.shape == (2, 2, 5)
    assert out[0].abs().sum() > 0
    assert out[1, 0].abs().sum() > 0
    assert out[1, 1].abs().sum() == 0
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_positions_pads_to_whole_tokens():
    encoder = words()
    out = encoder.positions(
        torch.randn(1, SENSORS, 2 * DOWNSAMPLING + 5), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long)
    )
    assert out.shape == (1, 1, 5)


def test_positions_full_precision_trains_the_trunk():
    encoder = words(SpannedConfig(components=3, width=5, full_precision=True))
    x = torch.randn(1, SENSORS, 2 * DOWNSAMPLING, requires_grad=False)
    encoder.positions(x, torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long)).sum().backward()
    assert encoder.trunk.module.scale.grad is not None
    assert encoder.reduce[1].weight.grad is not None


def test_embed():
    encoder = words().eval()
    x = torch.randn(3, SENSORS, 2 * DOWNSAMPLING)
    embedded = encoder.embed(x)
    assert embedded.shape == (3, 5)
    assert torch.allclose(embedded[1:2], encoder.embed(x[1:2]), atol=1e-5)


def test_weights():
    start = torch.tensor([[0, 2 * DOWNSAMPLING + 1]])
    weights = SpanWords.weights(start, 6)
    assert weights.shape == (1, 2, 6)
    assert weights.sum(-1)[0].tolist() == pytest.approx([1.0, 1.0])
    for at, first in enumerate(start[0].tolist()):
        covered = range(first // DOWNSAMPLING, (first + WORD - 1) // DOWNSAMPLING + 1)
        expected = torch.zeros(6)
        expected[list(covered)] = 1 / len(covered)
        assert torch.allclose(weights[0, at], expected)

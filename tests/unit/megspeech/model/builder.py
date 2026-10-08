from pathlib import Path

import pytest
from torch import nn

from core.paths import DataRoot
from megspeech.model.builder import WordBuilder
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.words.conv import ConvConfig, ConvWords
from megspeech.model.words.experts import ExpertWords
from megspeech.model.words.megxl import POOLED, RESOLVED
from megspeech.model.words.spanned import SpannedConfig, SpanWords
from megspeech.model.words.trunk import Geometry, Trunk
from tests.unit.megspeech.model.words.trunk import CHANNELS, Features


def test_build() -> None:
    conv = ConvConfig(hidden=8, dilations=(1,))
    config = ContextConfig(conv=conv)
    assert isinstance(WordBuilder.build(Encoder.CONV, config), ConvWords)
    assert WordBuilder.build(Encoder.CACHED, config).width == POOLED
    assert WordBuilder.build(Encoder.RESOLVED, config).width == RESOLVED
    assert WordBuilder.build(Encoder.TRAINED, config).width == conv.width
    narrow = WordBuilder.build(Encoder.CONV, config, 5)
    assert isinstance(narrow, ConvWords)
    assert isinstance(narrow.spatial, nn.Conv1d)
    assert narrow.spatial.in_channels == 5
    assert WordBuilder.build(Encoder.CACHED, config, 5).width == POOLED
    experts = ContextConfig(conv=conv, encoder=Encoder.EXPERTS, starts=(0, 10), lengths=(50, 10))
    assert isinstance(WordBuilder.build(Encoder.EXPERTS, experts, 5), ExpertWords)
    assert WordBuilder.build(Encoder.EXPERTS, experts, 5).width == 2 * conv.width


def test_build_spanned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    monkeypatch.setattr(Trunk, "pretrained", classmethod(lambda _cls, _root: trunk))
    monkeypatch.setattr(DataRoot, "from_config", classmethod(lambda _cls: DataRoot(tmp_path)))
    config = ContextConfig(encoder=Encoder.SPANNED, spanned=SpannedConfig(components=3, width=7))
    built = WordBuilder.build(Encoder.SPANNED, config, 2)
    assert isinstance(built, SpanWords)
    assert (built.trunk is trunk, built.width) == (True, 7)

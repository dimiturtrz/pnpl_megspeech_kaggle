from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.megxl import FeatureWords, MegxlWords
from megspeech.model.words.trunk import LATENT, Geometry, Trunk
from tests.unit.megspeech.model.words.trunk import CHANNELS, Features

FEATURES = torch.arange(2.0 * 3 * 4).reshape(2, 3, 4)


def words(*, instance: bool = False) -> MegxlWords:
    return MegxlWords(Trunk(Features(), Geometry.of(CHANNELS)), ConvConfig(instance=instance))


def test_pretrained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    monkeypatch.setattr(Trunk, "pretrained", classmethod(lambda _cls, _root: trunk))
    assert MegxlWords.pretrained(DataRoot(tmp_path), ConvConfig()).trunk is trunk


def test_megxl_words_embed():
    embedded = words().embed(torch.arange(16.0).reshape(1, 2, 8))
    assert embedded.shape == (1, 2 * LATENT)
    assert embedded[0, 0].item() < embedded[0, LATENT].item()
    x = 3.0 + torch.arange(16.0).reshape(1, 2, 8)
    assert torch.allclose(words(instance=True).normalise(x, x.ge(0)[:, :1]).mean(-1), torch.zeros(1, 2), atol=1e-5)


def test_feature_words_embed():
    encoder = FeatureWords(ConvConfig(instance=True), 3 * 4)
    assert torch.equal(encoder.embed(FEATURES), FEATURES.reshape(2, 12))
    assert encoder.normalise(FEATURES, torch.ones(2, 1, 4, dtype=torch.bool)) is FEATURES

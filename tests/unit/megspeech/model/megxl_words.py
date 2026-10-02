from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.vocabulary import WORDS
from megspeech.model.megxl import LATENT, Geometry, Trunk
from megspeech.model.megxl_words import Words
from megspeech.model.wordnet import ConvConfig
from tests.unit.megspeech.model.megxl import CHANNELS, Features


def words(*, instance: bool = False) -> Words:
    return Words(Trunk(Features(), Geometry.of(CHANNELS)), ConvConfig(instance=instance))


def test_pretrained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trunk = Trunk(Features(), Geometry.of(CHANNELS))
    monkeypatch.setattr(Trunk, "pretrained", classmethod(lambda _cls, _root: trunk))
    assert Words.pretrained(DataRoot(tmp_path), ConvConfig()).trunk is trunk


def test_embed():
    x = torch.arange(16.0).reshape(1, 2, 8)
    embedded = words().embed(x)
    assert embedded.shape == (1, 2 * LATENT)
    assert embedded[0, 0].item() < embedded[0, LATENT].item()


def test_normalise():
    x = 3.0 + torch.arange(16.0).reshape(1, 2, 8)
    present = torch.ones(1, 1, 8, dtype=torch.bool)
    assert torch.equal(words().normalise(x, present), x)
    assert torch.allclose(words(instance=True).normalise(x, present).mean(-1), torch.zeros(1, 2), atol=1e-5)


def test_forward():
    assert words().forward(torch.ones(3, 2, 8)).shape == (3, WORDS)

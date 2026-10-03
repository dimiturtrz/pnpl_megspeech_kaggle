from pathlib import Path

import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.model.encoder import Encoder
from megspeech.model.megxl_words import POOLED, RESOLVED, Words
from megspeech.model.wordnet import ConvConfig, WordNet


def test_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = ConvConfig(hidden=8, dilations=(1,))
    assert isinstance(Encoder.CONV.build(config), WordNet)
    assert Encoder.CACHED.build(config).width == POOLED
    assert Encoder.RESOLVED.build(config).width == RESOLVED
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    built: list[tuple[Path, ConvConfig]] = []
    monkeypatch.setattr(
        Words, "pretrained", classmethod(lambda _cls, root, conv: built.append((root.foreign("x"), conv)))
    )
    Encoder.MEGXL.build(config)
    assert built == [(DataRoot(tmp_path).foreign("x"), config)]


def test_cached() -> None:
    assert [encoder for encoder in Encoder if encoder.cached] == [Encoder.CACHED, Encoder.RESOLVED]

from pathlib import Path

import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.model.cached import Cached
from megspeech.model.encoder import Encoder
from megspeech.model.megxl_words import Words
from megspeech.model.wordnet import ConvConfig, WordNet


def test_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = ConvConfig(hidden=8, dilations=(1,))
    assert isinstance(Encoder.CONV.build(config), WordNet)
    assert isinstance(Encoder.CACHED.build(config), Cached)
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    built: list[tuple[Path, ConvConfig]] = []
    monkeypatch.setattr(
        Words, "pretrained", classmethod(lambda _cls, root, conv: built.append((root.foreign("x"), conv)))
    )
    Encoder.MEGXL.build(config)
    assert built == [(DataRoot(tmp_path).foreign("x"), config)]

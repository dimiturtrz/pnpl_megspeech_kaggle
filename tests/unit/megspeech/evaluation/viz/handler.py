from pathlib import Path

import pytest

from megspeech.evaluation.viz.handler import Handler


@pytest.fixture
def handler(tmp_path: Path) -> Handler:
    unbound = Handler.__new__(Handler)
    unbound.web, unbound.data = tmp_path / "web", tmp_path / "data"
    return unbound


@pytest.mark.parametrize(
    ("path", "under", "rest"),
    [
        ("/data/meg/a.f32?v=1", "data", "meg/a.f32"),
        ("/data", "data", ""),
        ("/database.js", "web", "database.js"),
        ("/", "web", ""),
        ("/data/../../secret", "data", "secret"),
    ],
)
def test_translate_path(handler: Handler, tmp_path: Path, path: str, under: str, rest: str) -> None:
    assert Path(handler.translate_path(path)) == tmp_path / under / rest


@pytest.mark.parametrize(
    ("path", "mime"),
    [
        ("/data/meg/a.f32", "application/octet-stream"),
        ("/data/audio/a.wav", "audio/wav"),
        ("/data/manifest.json", "application/json"),
        ("/app.js", "text/javascript"),
        ("/index.html", "text/html"),
    ],
)
def test_guess_type(handler: Handler, path: str, mime: str) -> None:
    assert handler.guess_type(path) == mime

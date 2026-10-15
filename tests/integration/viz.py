"""Server -> Handler -> HTTP client: the viewer's two mounts answer over a real loopback socket."""

import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from megspeech.evaluation.viz.serve import ServeConfig, Server

INDEX = b"<!doctype html><title>viz</title>"
MANIFEST = b'{"name": "default"}'
MEG = b"\x00\x00\x80?"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[Server]:
    web, data = tmp_path / "web", tmp_path / "data"
    (data / "meg").mkdir(parents=True)
    web.mkdir()
    (web / "index.html").write_bytes(INDEX)
    (data / "x.json").write_bytes(MANIFEST)
    (data / "meg" / "a.f32").write_bytes(MEG)
    running = Server(ServeConfig(data=data, port=0, web=web))
    thread = threading.Thread(target=running.serve, daemon=True)
    thread.start()
    yield running
    running.close()
    thread.join()


@pytest.mark.parametrize(
    ("path", "body", "mime"),
    [
        ("", INDEX, "text/html"),
        ("data/x.json", MANIFEST, "application/json"),
        ("data/meg/a.f32", MEG, "application/octet-stream"),
    ],
)
def test_viewer_and_export_mounts(server: Server, path: str, body: bytes, mime: str) -> None:
    with urllib.request.urlopen(server.url() + path) as response:
        assert (response.read(), response.headers["Content-Type"]) == (body, mime)


def test_export_is_not_on_the_web_root(server: Server) -> None:
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(server.url() + "x.json")

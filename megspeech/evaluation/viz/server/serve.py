"""Serve the viewer: `web/` at `/`, one export under the data root's `viz/<name>/` at `/data/`."""

import logging
from dataclasses import dataclass
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from core.paths import DataRoot
from megspeech.evaluation.viz.server.handler import Handler

WEB = Path(__file__).resolve().parents[1] / "web"
"""The static viewer app, served at `/`."""

VIZ = "viz"
"""The data-root kind exports live under: `<data>/viz/<name>/`."""

HOST = "127.0.0.1"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServeConfig:
    """What to serve and where: the viewer app, one export directory, the port (0 picks a free one)."""

    data: Path
    port: int = 8000
    web: Path = WEB

    @classmethod
    def for_export(cls, name: str, port: int, root: DataRoot | None = None) -> "ServeConfig":
        """The config serving export `name` from the data root (paths.yaml unless `root` is given)."""
        return cls(data=(root or DataRoot.from_config()).kind(VIZ) / name, port=port)


class Server:
    """A bound threaded HTTP server for one `ServeConfig`; `serve` blocks until `close`, then frees the socket."""

    def __init__(self, config: ServeConfig) -> None:
        handler = partial(Handler, web=config.web, data=config.data)
        self._httpd = ThreadingHTTPServer((HOST, config.port), handler)
        self._config = config

    def url(self) -> str:
        """The address the server answers on, with the port it actually bound."""
        return f"http://{HOST}:{self._httpd.server_address[1]}/"

    def serve(self) -> None:
        """Serve until `close` is called from another thread (or the process is interrupted); release the socket."""
        log.info("serving %s (data %s) at %s", self._config.web, self._config.data, self.url())
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def close(self) -> None:
        """Stop `serve`, waiting for it to return. Call only once `serve` is running or about to run."""
        self._httpd.shutdown()

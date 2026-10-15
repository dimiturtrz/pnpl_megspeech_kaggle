"""Request handler for the viewer: two static roots behind one URL space."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket
from typing import override

DATA = "/data"
"""URL prefix the export is mounted at."""


class Handler(SimpleHTTPRequestHandler):
    """Static files from two roots: `/data/...` from the export, everything else from the viewer app."""

    extensions_map = {  # noqa: RUF012
        **SimpleHTTPRequestHandler.extensions_map,
        ".f32": "application/octet-stream",
        ".wav": "audio/wav",
        ".json": "application/json",
        ".js": "text/javascript",
        ".html": "text/html",
        ".css": "text/css",
    }

    def __init__(
        self, request: socket, client: tuple[str, int], server: ThreadingHTTPServer, *, web: Path, data: Path
    ) -> None:
        self.web = web
        self.data = data
        super().__init__(request, client, server, directory=str(web))

    @override
    def translate_path(self, path: str) -> str:
        """The file a URL path names: under the export for `/data/...`, under the viewer app otherwise."""
        mounted = path == DATA or path.startswith((f"{DATA}/", f"{DATA}?"))
        self.directory = str(self.data if mounted else self.web)
        return super().translate_path(path.removeprefix(DATA) if mounted else path)

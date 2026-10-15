import threading
from pathlib import Path

from core.paths import DataRoot
from megspeech.evaluation.viz.serve import WEB, ServeConfig, Server


def test_for_export(tmp_path: Path) -> None:
    config = ServeConfig.for_export("run7", 9000, DataRoot(tmp_path))
    assert config == ServeConfig(data=tmp_path / "viz" / "run7", port=9000, web=WEB)


def test_url(tmp_path: Path) -> None:
    server = Server(ServeConfig(data=tmp_path, port=0, web=tmp_path))
    stopper = threading.Thread(target=server.close)
    stopper.start()
    server.serve()
    assert server.url().startswith("http://127.0.0.1:")
    assert not server.url().endswith(":0/")


def test_serve(tmp_path: Path) -> None:
    server = Server(ServeConfig(data=tmp_path, port=0, web=tmp_path))
    stopper = threading.Thread(target=server.close)
    stopper.start()
    server.serve()
    stopper.join()
    assert not stopper.is_alive()


def test_close(tmp_path: Path) -> None:
    server = Server(ServeConfig(data=tmp_path, port=0, web=tmp_path))
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    server.close()
    thread.join(timeout=5)
    assert not thread.is_alive()

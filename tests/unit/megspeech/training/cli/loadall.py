import json
import logging
from pathlib import Path

import pytest

from core.paths import DataRoot
from megspeech.training.cli import loadall as module
from megspeech.training.cli.loadall import LoadAll, main
from megspeech.training.config import METRICS


def run(root: Path, name: str, record: object) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / METRICS).write_text(json.dumps(record))
    return directory


def archive(root: Path) -> LoadAll:
    run(root, "good", {"config": {"optim": {"epochs": 2}}})
    run(root, "bad", {"config": {"nowhere": 1}})
    run(root, "bare", {})
    (root / "empty").mkdir()
    return LoadAll(root)


def test_runs(tmp_path: Path) -> None:
    assert [path.name for path in archive(tmp_path).runs()] == ["bad", "bare", "good"]


def test_failure(tmp_path: Path) -> None:
    runs = {path.name: path for path in archive(tmp_path).runs()}
    assert LoadAll.failure(runs["good"]) is None
    assert "nowhere" in str(LoadAll.failure(runs["bad"]))
    assert str(LoadAll.failure(runs["bare"])).startswith("KeyError")


def test_failures(tmp_path: Path) -> None:
    assert sorted(archive(tmp_path).failures()) == ["bad", "bare"]


def test_report() -> None:
    assert LoadAll.report(5, [("a", "why")]) == ["loaded\t5", "a\twhy"]


def test_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    archive(models)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(DataRoot, "models", lambda _self: models)
    with caplog.at_level(logging.INFO):
        main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "loaded	1"
    assert [line.split("	")[0] for line in lines[1:]] == ["bad", "bare"]
    assert "1 loaded, 2 failing" in caplog.text

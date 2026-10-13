import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import AS_RECORDED
from megspeech.data.cache.kaggle import Track
from megspeech.evaluation.group import Group
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.probabilities import PROBS, SECTIONS, Probabilities
from megspeech.submission import __main__ as cli
from megspeech.submission import report as module
from megspeech.submission.config import ReportConfig
from megspeech.submission.report import HITS, SCORING, SUBMISSIONS, Report, Submission
from tests.unit.megspeech.evaluation.loaded import CPU, loaded, runs

RUN = "run"


def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scorings: list[Scoring]) -> None:
    """One loaded model behind `Loader.of`, and a Kaggle writer that records the adjustment its rows are written at."""
    every = runs(tmp_path)

    def of(group: Group, names: list[str]) -> LoadedModel:
        scorings.append(group.scoring)
        return replace(loaded(tmp_path), names=tuple(names))

    def write(self: module.KaggleWriter, _: LoadedModel, tau: float, path: Path) -> Path:
        path.write_text(f"{self.track} {tau}")
        return path

    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(module.Loader, "of", staticmethod(of))
    monkeypatch.setattr(module.KaggleWriter, "write", write)


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub(tmp_path, monkeypatch, [])
    for track in Track:
        out = tmp_path / track
        written = Submission(track).write(replace(loaded(tmp_path), names=(RUN,)), runs(tmp_path), out)
        assert written == out / SUBMISSIONS[track]
        assert (out / HITS[track]).is_file()
        assert json.loads((out / "metrics.json").read_text())["config"]["scoring"] == json.loads(
            json.dumps(asdict(SCORING))
        )


def test_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Asking for nothing writes nothing; each track lands under models/<name>-<track>, aligned for Broad alone."""
    scorings: list[Scoring] = []
    stub(tmp_path, monkeypatch, scorings)
    root = DataRoot(tmp_path)
    Report(ReportConfig(), root, RUN, CPU, AS_RECORDED).run()
    assert scorings == [] and not list(root.models().glob(f"{RUN}-*"))
    config = ReportConfig(local=True, tracks=(Track.DEEP, Track.BROAD), aligned=True)
    with caplog.at_level(logging.INFO):
        Report(config, root, RUN, CPU, AS_RECORDED).run()
    assert [s.aligned for s in scorings] == [False, False, True]
    assert (root.models() / f"{RUN}-deep" / SUBMISSIONS[Track.DEEP]).is_file()
    assert (root.models() / f"{RUN}-broad" / SUBMISSIONS[Track.BROAD]).is_file()
    assert "run val tau" in caplog.text and "run local test tau" in caplog.text


def test_run_probabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asked to, the local report leaves both partitions' per-word probabilities next to the model."""
    stub(tmp_path, monkeypatch, [])
    root = DataRoot(tmp_path)
    (root.models() / RUN).mkdir(parents=True)
    Report(ReportConfig(local=True), root, RUN, CPU, AS_RECORDED, probabilities=True).run()
    sections = {key.split("/")[0].split("#")[0] for key in Probabilities.stored(root.models() / RUN / PROBS)}
    assert sections == set(SECTIONS.values())


def test_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    root = DataRoot(tmp_path)
    monkeypatch.setattr(module.Loader, "of", staticmethod(lambda *args: seen.append(args) or loaded(tmp_path)))
    geometry = AS_RECORDED
    Report(ReportConfig(), root, RUN, torch.device("cpu"), geometry).loaded(module.SCORING)
    assert seen == [(Group(root, module.SCORING, torch.device("cpu"), geometry), [RUN])]


def test_in_process_equals_separate_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The in-process reports write byte-for-byte what the `megspeech.submission` command writes for the same model."""
    stub(tmp_path, monkeypatch, [])
    root = DataRoot(tmp_path)
    monkeypatch.setattr(cli.DataRoot, "from_config", classmethod(lambda _: root))
    monkeypatch.setattr(cli.torch, "device", lambda _: CPU)
    command: dict[Track, list[str]] = {Track.DEEP: [], Track.BROAD: ["--aligned"]}
    texts = []
    for track, extra in command.items():
        argv = ["submission", RUN, "--out", f"{RUN}-{track}-cli", "--track", track, *extra]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
    with caplog.at_level(logging.INFO):
        monkeypatch.setattr(sys, "argv", ["submission", RUN, "--out", "x", "--local"])
        cli.main()
        texts.append([r.getMessage() for r in caplog.records if r.getMessage().startswith(RUN)])
        caplog.clear()
        Report(
            ReportConfig(local=True, tracks=(Track.DEEP, Track.BROAD), aligned=True), root, RUN, CPU, AS_RECORDED
        ).run()
        texts.append([r.getMessage() for r in caplog.records if r.getMessage().startswith(RUN)])
    assert texts[0] == texts[1] and texts[0]
    for track in Track:
        separate, inline = root.models() / f"{RUN}-{track}-cli", root.models() / f"{RUN}-{track}"
        files = sorted(path.name for path in separate.iterdir())
        assert files == sorted(path.name for path in inline.iterdir())
        assert all((separate / name).read_bytes() == (inline / name).read_bytes() for name in files)

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.paths import DataRoot
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.evaluation.labelweighted import Shrinkage
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.input.subjects import Reading, Unseen
from megspeech.submission import __main__ as module
from megspeech.submission import report
from megspeech.submission.__main__ import main
from megspeech.submission.report import HITS, SUBMISSIONS
from tests.unit.megspeech.evaluation.loaded import loaded, runs


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Each track's ruler selects the adjustment its Kaggle test rows are written at, under models/<out>."""
    every = runs(tmp_path)
    scorings: list[Scoring] = []
    written: list[tuple[Track, float, Path]] = []

    def of(_: DataRoot, names: list[str], scoring: Scoring, __: object) -> LoadedModel:
        scorings.append(scoring)
        return replace(loaded(tmp_path), names=tuple(names))

    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(module.Loader, "of", staticmethod(of))
    monkeypatch.setattr(
        report.KaggleWriter, "write", lambda self, _, tau, path: written.append((self.track, tau, path)) or path
    )
    for track in Track:
        monkeypatch.setattr(
            sys, "argv", ["submission", "a", "b", "--out", "both", "--track", track, "--reading", "unseen"]
        )
        with caplog.at_level(logging.INFO, logger=module.__name__):
            main()
    out = DataRoot(tmp_path).models() / "both"
    metrics = json.loads((out / METRICS).read_text())
    assert (metrics["members"], metrics["config"]["data"]["window"]) == (["a", "b"], loaded(tmp_path).data.window)
    assert [(track, path) for track, _, path in written] == [(track, out / SUBMISSIONS[track]) for track in Track]
    assert {(s.reading, s.fallback.unseen) for s in scorings} == {(Reading.UNSEEN, Unseen.DEEP)}
    assert f"wrote {out / SUBMISSIONS[Track.BROAD]}" in caplog.text
    assert all((out / HITS[track]).exists() for track in Track)


def test_main_labelweights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--labelweights K BETA writes through the per-label blend under models/<out>, and refuses the Deep track."""
    written: list[tuple[Shrinkage, Path]] = []

    class Stub:
        def write(self, _: Path, out: Path) -> Path:
            written.append((shrinkage, out))
            return out

    shrinkage = Shrinkage(5.0, 10.0)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(
        module.LabelWeighted, "read", staticmethod(lambda *args: Stub() if args[-1] == shrinkage else None)
    )
    argv = ["submission", "a", "b", "--out", "lw", "--labelweights", "5", "10", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", [*argv, "--track", "broad"])
    main()
    assert written == [(shrinkage, DataRoot(tmp_path).models() / "lw")]
    monkeypatch.setattr(sys, "argv", [*argv, "--track", "deep"])
    with pytest.raises(SystemExit):
        main()

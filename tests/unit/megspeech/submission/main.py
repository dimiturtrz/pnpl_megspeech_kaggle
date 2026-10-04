import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.paths import DataRoot
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.evaluation.loaded import LoadedModel, Scoring
from megspeech.model.input.subjects import Reading, Unseen
from megspeech.submission import __main__ as module
from megspeech.submission.__main__ import HITS, SUBMISSIONS, main
from tests.unit.megspeech.evaluation.loaded import loaded, runs


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Each track's ruler selects the adjustment its Kaggle test rows are written at, under models/<out>."""
    every = runs(tmp_path)
    scorings: list[Scoring] = []
    written: list[tuple[Track, float, Path]] = []

    def of(_: type[LoadedModel], __: DataRoot, names: list[str], scoring: Scoring, ___: object) -> LoadedModel:
        scorings.append(scoring)
        return replace(loaded(tmp_path), names=tuple(names))

    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(module.LoadedModel, "of", classmethod(of))
    monkeypatch.setattr(
        module.KaggleWriter, "write", lambda self, _, tau, path: written.append((self.track, tau, path)) or path
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

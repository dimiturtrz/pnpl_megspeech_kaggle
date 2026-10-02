import logging
import sys
from pathlib import Path

import pytest

from core.paths import DataRoot
from megspeech.data.holdout import Track
from megspeech.evaluation import broad as ruler
from megspeech.evaluation.broad import Reading
from megspeech.submission import broad as module
from megspeech.submission.broad import SUBMISSION, main
from tests.unit.megspeech.evaluation.pooling import pooling, runs


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The ruler's adjustment is the one the Broad track's rows are written at, into the model's directory."""
    every = runs(tmp_path)
    written: list[tuple[Track, Reading, float, Path]] = []

    def submit(self: module.Submitter, _: object, __: object, ___: object, tau: float, path: Path) -> Path:
        written.append((self.track, self.reading, tau, path))
        return path

    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(ruler.Pooling, "of", classmethod(lambda *_: pooling(tmp_path)))
    monkeypatch.setattr(module.Submitter, "submit", submit)
    monkeypatch.setattr(sys, "argv", ["broad", "m", "--reading", "unseen"])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        main()
    path = DataRoot(tmp_path).models() / "m" / SUBMISSION
    assert written == [(Track.BROAD, Reading.UNSEEN, 0.0, path)]
    assert f"wrote {path}" in caplog.text

import json
from pathlib import Path

import pytest

from megspeech.data.config import METRICS
from megspeech.evaluation.scoring import config as module
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.input.lowpass import LowPass
from tests.unit.megspeech.evaluation.loaded.model import CPU, Favourite

RATE = 50.0
CUTOFF = 12.0


def test_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ensemble is returned as is, or behind a low-pass when the scoring asks for one."""
    loaded: list[list[Path]] = []
    favourite = Favourite()
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda _c, paths, _u: loaded.append(paths) or favourite))
    assert Scoring().network([Path("a")], RATE, CPU) is favourite
    assert loaded == [[Path("a") / "model.pt"]]
    filtered = Scoring(lowpass=CUTOFF).network([Path("a")], RATE, CPU)
    assert isinstance(filtered, LowPass)
    assert (filtered.model, filtered.rate, filtered.cutoff) == (favourite, RATE, CUTOFF)


def test_members(tmp_path: Path) -> None:
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / METRICS).write_text(json.dumps({"members": ["a", "b"]}))
    assert Scoring.members(tmp_path, ["ab", "c"]) == [tmp_path / "a", tmp_path / "b", tmp_path / "c"]


def test_word_share() -> None:
    assert Scoring().word_share == 0.0
    assert Scoring(word_share=1.0).word_share == 1.0
    with pytest.raises(ValueError, match="word_share"):
        Scoring(word_share=1.5)

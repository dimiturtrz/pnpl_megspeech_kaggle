from pathlib import Path

from megspeech.evaluation.rulers.base import TrackRuler
from megspeech.evaluation.rulers.broad import BroadRuler
from megspeech.evaluation.rulers.deep import DeepRuler
from megspeech.evaluation.scoring.score import ADJUSTMENTS
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded import loaded, runs

TRACKS: tuple[type[TrackRuler], ...] = (DeepRuler, BroadRuler)


def rulers(tmp_path: Path) -> dict[type[TrackRuler], TrackRuler]:
    return {track: track(loaded(tmp_path), runs(tmp_path)) for track in TRACKS}


def test_select(tmp_path: Path) -> None:
    """Every track's ruler selects from the model and runs it holds."""
    selected = [ruler.select().tau for ruler in rulers(tmp_path).values()]
    assert set(selected) <= set(ADJUSTMENTS)
    assert len(selected) == len(TRACKS)


def test_graded(tmp_path: Path) -> None:
    """Every track's ruler grades, at a given tau, the domains it counts."""
    graded = {track: set(ruler.graded(0.5).domains) for track, ruler in rulers(tmp_path).items()}
    assert RULER in graded[DeepRuler]
    assert graded[BroadRuler] == {"1"}

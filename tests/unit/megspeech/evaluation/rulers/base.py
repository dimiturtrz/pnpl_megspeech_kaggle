import inspect

from megspeech.evaluation.rulers.base import TrackRuler
from megspeech.evaluation.rulers.broad import BroadRuler
from megspeech.evaluation.rulers.deep import DeepRuler


def test_select() -> None:
    """Every track's ruler, called on an instance, takes the protocol's arguments."""
    expected = list(inspect.signature(TrackRuler.select).parameters)[1:]
    assert all(list(inspect.signature(r().select).parameters) == expected for r in (DeepRuler, BroadRuler))

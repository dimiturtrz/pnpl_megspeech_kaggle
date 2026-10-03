"""What a track's ruler does: score a loaded model the way that track's leaderboard would, and pick its adjustment."""

from typing import Protocol

from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.scoring.score import Epoch, Score


class TrackRuler(Protocol):
    def select(self, loaded: LoadedModel, runs: list[Run]) -> Epoch | Score:
        """Score `loaded` on `runs`, log it, and return the record whose `tau` its submission is written at."""
        ...

    def graded(self, loaded: LoadedModel, runs: list[Run], tau: float) -> Graded:
        """`loaded`'s per-word hits at `tau` on every domain `select` counts, to compare two models paired."""
        ...

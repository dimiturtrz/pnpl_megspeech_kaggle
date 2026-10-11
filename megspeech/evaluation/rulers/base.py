"""What a track's ruler does: score a loaded model the way that track's leaderboard would, and pick its adjustment."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.scoring.score import Epoch, Score


@dataclass(frozen=True)
class TrackRuler(ABC):
    """`loaded` scored on `runs` as one track's leaderboard would score it."""

    loaded: LoadedModel
    runs: list[Run]

    @abstractmethod
    def select(self) -> Epoch | Score:
        """Score `loaded` on `runs`, log it, and return the record whose `tau` its submission is written at."""

    @abstractmethod
    def graded(self, tau: float) -> Graded:
        """`loaded`'s per-word hits at `tau` on every domain `select` counts, to compare two models paired."""

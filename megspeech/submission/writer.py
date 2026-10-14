"""A loaded model's Kaggle test predictions for a track, written as a Kaggle submission."""

from dataclasses import dataclass
from pathlib import Path

from megspeech.data.cache.kaggle import Track
from megspeech.evaluation.loaded.exported import ExportedModel


@dataclass(frozen=True)
class KaggleWriter:
    track: Track

    def write(self, loaded: ExportedModel, tau: float, path: Path) -> Path:
        """`ExportedModel.probabilities` for every row of the track's Kaggle test, each read from its own subject's MEG
        and logit-adjusted by `tau`; the secondary vocabulary uniform. Refused on Deep for a model that trained on
        `DataConfig.adapt`: that session's text is subject 0's local-test text, and Deep stays excluded (bead 5f3)."""
        if self.track is Track.DEEP and loaded.adapted_on:
            raise ValueError(f"a model adapted on session {loaded.adapted_on} cannot be submitted to the Deep track")
        return self.track.write(loaded.root.raw(), loaded.kaggle_probabilities(self.track, tau), path)

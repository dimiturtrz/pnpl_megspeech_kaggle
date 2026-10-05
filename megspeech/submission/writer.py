"""A loaded model's Kaggle test predictions for a track, written as a Kaggle submission."""

from dataclasses import dataclass
from pathlib import Path

from megspeech.data.cache.kaggle import Track
from megspeech.evaluation.loaded import LoadedModel


@dataclass(frozen=True)
class KaggleWriter:
    track: Track

    def write(self, loaded: LoadedModel, tau: float, path: Path) -> Path:
        """`LoadedModel.probabilities` for every row of the track's Kaggle test, each read from its own subject's MEG
        and logit-adjusted by `tau`; the secondary vocabulary uniform."""
        return self.track.write(loaded.root.raw(), loaded.probabilities(loaded.kaggle(self.track), tau), path)

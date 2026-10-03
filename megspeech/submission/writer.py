"""A loaded model's holdout predictions for a track, written as a Kaggle submission."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pnpl.competition import write_submission

from megspeech.data.batching.split import HOLDOUT_FEATURES, Split
from megspeech.data.cache.holdout import Track
from megspeech.data.labels.vocabulary import SECONDARY
from megspeech.evaluation.loaded import LoadedModel


@dataclass(frozen=True)
class HoldoutWriter:
    track: Track

    def write(self, loaded: LoadedModel, tau: float, path: Path) -> Path:
        """`LoadedModel.probabilities` for every row of the track's holdout, each read from its own subject's MEG and
        logit-adjusted by `tau`; the secondary vocabulary uniform."""
        holdout = self.track.holdout(loaded.root.raw())
        windows = loaded.data.holdout(loaded.root, self.track, loaded.evaluator.device, loaded.alignment)
        split = loaded.read(f"{HOLDOUT_FEATURES}-{self.track}", Split.from_holdout(holdout, windows))
        primary = loaded.probabilities(split, tau)
        secondary = np.full((len(primary), len(SECONDARY.words)), 1 / len(SECONDARY.words))
        return write_submission(path, holdout.indices, primary, secondary)

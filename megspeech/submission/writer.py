"""A loaded model's Kaggle test predictions for a track, written as a Kaggle submission."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pnpl.competition import write_submission

from megspeech.data.batching.split import KAGGLE_FEATURES, Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import SECONDARY
from megspeech.evaluation.loaded import LoadedModel


@dataclass(frozen=True)
class KaggleWriter:
    track: Track

    def write(self, loaded: LoadedModel, tau: float, path: Path) -> Path:
        """`LoadedModel.probabilities` for every row of the track's Kaggle test, each read from its own subject's MEG
        and logit-adjusted by `tau`; the secondary vocabulary uniform."""
        kaggle = self.track.kaggle(loaded.root.raw())
        windows = Corpus.kaggle(loaded.root, loaded.data, self.track, loaded.device, loaded.geometry)
        split = loaded.read(f"{KAGGLE_FEATURES}-{self.track}", Split.from_kaggle(kaggle, windows))
        primary = loaded.probabilities(split, tau)
        secondary = np.full((len(primary), len(SECONDARY.words)), 1 / len(SECONDARY.words))
        return write_submission(path, kaggle.indices, primary, secondary)

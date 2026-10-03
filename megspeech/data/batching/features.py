"""Per-window features a frozen encoder computed once, read in place of the windows' MEG.

A frozen pretrained trunk maps each window to the same features every epoch, so they are computed once
(`megspeech.training.features`) and training reads them at the cost of a lookup. Each split's features are saved
in its own words' order, so they stand in for its windows row for row.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows

FEATURES = "features"
TRAIN = "train"
HOLDOUT_FEATURES = "holdout"
"""The names of the training split's features and the prefix of a track's holdout's, `holdout-<track>`; a
validation domain's are its own name."""


@dataclass(frozen=True)
class Features:
    """The features saved under `directory`, one file per split; no directory reads every split's MEG."""

    directory: Path | None = None

    @classmethod
    def under(cls, root: DataRoot, name: str) -> "Features":
        return cls(root.kind("models") / FEATURES / name if name else None)

    def path(self, split: str) -> Path:
        if self.directory is None:
            raise ValueError("no features named")
        return self.directory / f"{split}.pt"

    def save(self, split: str, features: Float[Tensor, "n c *k"]) -> Path:
        path = self.path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features.cpu(), path)
        return path

    def read(self, split: str, windows: Windows) -> Windows:
        """`windows` itself, or the saved features of `split` as its windows, uncentred: channels by
        features each, one feature when saved flat."""
        if self.directory is None:
            return windows
        features = torch.load(self.path(split), map_location=windows.device, weights_only=True)
        if len(features) != len(windows):
            raise ValueError(f"{len(features)} {split} features for {len(windows)} windows")
        return Windows.stacked(features.reshape(len(features), features.shape[1], -1), centred=False)

    def split(self, name: str, split: Split) -> Split:
        """`split` reading `name`'s features as its windows."""
        return replace(split, windows=self.read(name, split.windows))

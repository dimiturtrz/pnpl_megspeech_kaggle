"""The Neuromag-306 helmet the viewer groups MEG rows by: each channel's name and kind, and MNE's lobe selections."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.paths import DataRoot
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.sources.masc import Neuromag

REGIONS = (
    "Left-temporal",
    "Right-temporal",
    "Left-frontal",
    "Right-frontal",
    "Left-parietal",
    "Right-parietal",
    "Left-occipital",
    "Right-occipital",
    "Vertex",
)
"""MNE's Vectorview selections; a channel near a border sits in two."""


class Layout(Protocol):
    def describe(self, root: DataRoot) -> dict[str, object]: ...


@dataclass(frozen=True)
class Helmet:
    """The layout of the recorded channels, in the exported windows' row order."""

    def describe(self, root: DataRoot) -> dict[str, object]:  # devtools-ignore: test-mirror
        return self.layout(Neuromag.of(NEUROMAG.home(root)))

    @staticmethod
    def layout(helmet: Neuromag) -> dict[str, object]:
        """Names, kinds (`mag`, `grad`: each location holds a magnetometer and two planar gradiometers) and each
        region's channel indices."""
        return {
            "names": helmet.names,
            "kinds": helmet.types,
            "regions": {region: np.flatnonzero(helmet.within((region,))).tolist() for region in REGIONS},
        }

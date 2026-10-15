"""The Neuromag-306 helmet the viewer groups MEG rows by: each channel's name and kind, and MNE's lobe selections."""

from dataclasses import dataclass
from typing import Protocol

import mne
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
LAYOUT = "Vectorview-all"
"""MNE's flat Neuromag-306 layout: each channel's box on the unit square, nose up."""


class Layout(Protocol):
    def describe(self, root: DataRoot) -> dict[str, object]: ...


@dataclass(frozen=True)
class Helmet:
    """The layout of the recorded channels, in the exported windows' row order."""

    def describe(self, root: DataRoot) -> dict[str, object]:  # devtools-ignore: test-mirror
        return self.layout(Neuromag.of(NEUROMAG.home(root)))

    @staticmethod
    def layout(helmet: Neuromag) -> dict[str, object]:
        """Names, kinds (`mag`, `grad`: each location holds a magnetometer and two planar gradiometers), each
        channel's flat helmet position (`plane`) and each region's channel indices."""
        return {
            "names": helmet.names,
            "kinds": helmet.types,
            "xy": Helmet.plane(helmet.names),
            "regions": {region: np.flatnonzero(helmet.within((region,))).tolist() for region in REGIONS},
        }

    @staticmethod
    def plane(names: list[str]) -> list[list[float] | None]:
        """Each channel's box centre in `LAYOUT`, x right and y toward the nose on the unit square; None off it."""
        layout = mne.channels.read_layout(LAYOUT)
        centre = layout.pos[:, :2] + layout.pos[:, 2:] / 2
        at = {name.replace(" ", ""): np.round(xy, 4).tolist() for name, xy in zip(layout.names, centre, strict=True)}
        return [at.get(name) for name in names]

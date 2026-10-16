"""Regional spatial dropout (Défossez et al. 2023): every training window loses all sensors near one random sensor.

Dropping single sensors removes little after SSS, whose rank-~80 field each sensor's neighbours reconstruct; a whole
patch removes what no neighbour holds, so the model has to read the field from more than one region. Défossez drop
within 0.2 of a unit-square layout, and removing it cost them 1.7 points on story listening (megspeech-7gf, augmentation
survey #2).
"""

from dataclasses import dataclass
from typing import Self

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.sources.meg_masc import Neuromag


@dataclass(frozen=True)
class Regional:
    near: Bool[Tensor, "c c"]
    """Whether sensor j lies within the radius of sensor i; all False drops nothing."""

    @classmethod
    def of(cls, share: float, root: DataRoot, device: torch.device) -> Self:
        """Patches of radius `share` of the array's widest extent, over the Neuromag-306 positions; none at 0."""
        if not share:
            return cls.off()
        positions = torch.from_numpy(Neuromag.of(NEUROMAG.home(root)).positions).float().to(device)
        return cls.within(positions, share)

    @classmethod
    def off(cls) -> Self:
        return cls(torch.zeros(0, 0, dtype=torch.bool))

    @classmethod
    def within(cls, positions: Float[Tensor, "c 3"], share: float) -> Self:
        extent = (positions.amax(0) - positions.amin(0)).max()
        return cls(torch.cdist(positions, positions) <= share * extent)

    def __call__(self, x: Float[Tensor, "n c t"]) -> Float[Tensor, "n c t"]:
        if not self.near.numel():
            return x
        centre = torch.randint(0, self.near.shape[0], (x.shape[0],), device=x.device)
        return x * ~self.near[centre][..., None]

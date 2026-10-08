"""Each sensor's word-locked evoked field on subject 0's training words, and whether its early (M50-M100) peak sits
over the temporal lobes, as the acoustic expert's spatial prior assumes (megspeech-7gf.77).

    uv run python -m megspeech.evaluation.topography
"""

import logging
import math
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.masc import Neuromag

LOG = logging.getLogger(__name__)

READING = DataConfig(window=0.5)
LEAD = 0.2
"""Seconds read before each onset and centred on: with `READING`'s window, the auditory evoked field's whole first
half-second."""
EARLY = (0.05, 0.15)
"""Seconds after onset of the auditory M50-M100, the response of primary auditory cortex."""
CHUNK = 4096
TOP = 20
STEP = 0.02
TEMPORAL = ("Left-temporal", "Right-temporal")


@dataclass(frozen=True)
class Evoked:
    """The mean and spread over `count` words of each sensor's window, sample by sample."""

    mean: Float[Tensor, "c s"]
    spread: Float[Tensor, "c s"]
    count: int

    @classmethod
    def of(cls, windows: Windows, chunk: int = CHUNK) -> Self:
        parts = torch.arange(len(windows), device=windows.device).split(chunk)
        total = torch.stack([windows(part).float().sum(0) for part in parts]).sum(0)
        squares = torch.stack([windows(part).float().square().sum(0) for part in parts]).sum(0)
        mean = total / len(windows)
        return cls(mean, (squares / len(windows) - mean.square()).clamp_min(0).sqrt(), len(windows))

    def t(self) -> Float[Tensor, "c s"]:
        """Each sample's mean in standard errors: unit-free, so magnetometers and gradiometers compare."""
        return self.mean / (self.spread / math.sqrt(self.count)).clamp_min(torch.finfo(torch.float32).tiny)

    def peaks(self, start: int, stop: int) -> Float[Tensor, " c"]:
        """Each sensor's largest |t| over samples `start` to `stop`."""
        return self.t()[:, start:stop].abs().amax(1)


def main() -> None:
    """Log the sensors with the strongest early evoked peak, the temporal share of them against chance, and the
    field's strength over time."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    train = Corpus.load(READING, root, torch.device("cuda")).train
    onset = READING.to_samples(LEAD)
    windows = replace(train.windows, length=READING.samples + onset, offset=-onset, baseline=onset)
    evoked, helmet = Evoked.of(windows), Neuromag.of(NEUROMAG.home(root))
    positions = helmet.positions
    peaks = evoked.peaks(*(onset + READING.to_samples(edge) for edge in EARLY)).cpu().numpy()
    temporal, order = helmet.within(TEMPORAL), np.argsort(-peaks)
    for i in order[:TOP]:
        LOG.info(
            "%s %s t=%.1f at %s",
            helmet.names[i],
            "mag" if helmet.names[i].endswith("1") else "grad",
            peaks[i],
            np.round(positions[i], 3),
        )
    left = positions[:, 0] < 0
    LOG.info(
        "temporal %d of top %d vs %.1f by chance (%d of %d sensors); left %d right %d; n=%d words",
        temporal[order[:TOP]].sum(),
        TOP,
        TOP * temporal.mean(),
        temporal.sum(),
        len(temporal),
        left[order[:TOP]].sum(),
        (~left)[order[:TOP]].sum(),
        evoked.count,
    )
    strength = evoked.t().square().mean(0).cpu().numpy()
    for sample in range(0, len(strength), READING.to_samples(STEP)):
        LOG.info("%+.2f s mean t^2 %.1f", (sample - onset) / READING.rate, strength[sample])


if __name__ == "__main__":
    main()

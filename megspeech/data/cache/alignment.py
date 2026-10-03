"""Euclidean alignment (He & Wu 2019): another subject's run re-coloured from its own sensor covariance to subject 0's.

A model trained on subject 0 alone meets every other subject through a different head and sensor geometry; whitening
each run by its own covariance and colouring it by subject 0's moves that geometry back onto the one it learned.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

GATE = 8.0
"""Interquartile ranges from its median at which a sample on any channel is an artifact the covariance leaves out:
subject 0's training runs reach 35-40 at the 99.9th percentile, the other subjects' runs and every holdout 3-4."""
FLOOR = 1e-4
"""Eigenvalues below this fraction of the largest are the Maxwell filter's null space (rank ~75 of 306): projected
out, never inverted."""
STRIDE = 2
"""Every how many samples the reference reads of each subject-0 run; its covariance needs far fewer than a run has."""
EVERY = 4
"""Every how many of subject 0's training runs the reference reads: 39 of 156 still span every book and session."""
QUANTILE_SAMPLE = 20000
REFERENCE = "sub-0_"
"""The prefix of subject 0's run names in both the training cache and the holdout cache (`subj00_`)."""
UNALIGNED = (REFERENCE, "subj00_")


@dataclass(frozen=True)
class Alignment:
    """Subject 0's per-channel median and interquartile range, and the square root of its sensor covariance in those
    units: what every aligned run is re-coloured to."""

    median: Float[Tensor, "c 1"]
    spread: Float[Tensor, "c 1"]
    root: Float[Tensor, "c c"]

    @classmethod
    def of(cls, runs: Sequence[Float[np.ndarray, "c s"]], device: torch.device | str) -> "Alignment":
        """Subject 0's reference from its training `runs`: the median of their medians and ranges, the mean of their
        covariances."""
        signals = [torch.from_numpy(np.asarray(run[:, ::STRIDE], dtype=np.float32)).to(device) for run in runs]
        medians, spreads = zip(*(cls.robust(signal) for signal in signals), strict=True)
        median = torch.stack(medians).median(0).values
        spread = torch.stack(spreads).median(0).values
        covariance = torch.stack([cls.covariance((s - m) / spread) for s, m in zip(signals, medians, strict=True)])
        return cls(median, spread, cls.power(covariance.mean(0), 0.5))

    @staticmethod
    def robust(x: Float[Tensor, "c s"]) -> tuple[Float[Tensor, "c 1"], Float[Tensor, "c 1"]]:
        """Each channel's median and interquartile range, the range floored above zero."""
        sample = x[:, :: max(1, x.shape[1] // QUANTILE_SAMPLE)].float()
        levels = torch.tensor([0.25, 0.5, 0.75], device=x.device)
        low, median, high = torch.quantile(sample, levels, dim=1, keepdim=True)
        return median, (high - low).clamp_min(torch.finfo(torch.float32).tiny)

    @staticmethod
    def covariance(z: Float[Tensor, "c s"]) -> Float[Tensor, "c c"]:
        """The sensor covariance of `z`'s samples within `GATE` on every channel, in float64."""
        kept = z[:, z.abs().amax(0) < GATE].double()
        centred = kept - kept.mean(1, keepdim=True)
        return centred @ centred.T / max(1, centred.shape[1])

    @staticmethod
    def power(covariance: Float[Tensor, "c c"], exponent: float) -> Float[Tensor, "c c"]:
        """`covariance` to `exponent` on its eigenvalues above `FLOOR` of the largest; zero on the rest."""
        values, vectors = torch.linalg.eigh(covariance)
        kept = values > values.max() * FLOOR
        scaled = torch.where(kept, values.clamp_min(values.max() * FLOOR) ** exponent, torch.zeros_like(values))
        return (vectors * scaled) @ vectors.T

    @staticmethod
    def applies(name: str) -> bool:
        """Whether run `name` is another subject's: subject 0's own runs are the reference and stay as recorded."""
        return not name.startswith(UNALIGNED)

    @torch.no_grad()
    def align(self, run: Float[Tensor, "c s"]) -> None:
        """`run` in place, in subject 0's units, whitened by its own covariance and coloured by subject 0's."""
        x = run.float()
        median, _ = self.robust(x)
        z = (x - median) / self.spread
        recolour = (self.root @ self.power(self.covariance(z), -0.5)).float()
        run.copy_((recolour @ z) * self.spread + self.median)

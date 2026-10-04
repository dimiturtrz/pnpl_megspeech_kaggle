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
subject 0's training runs reach 35-40 at the 99.9th percentile, the other subjects' runs and every Kaggle test
archive 3-4."""
FLOOR = 1e-4
"""Eigenvalues below this fraction of the largest are the Maxwell filter's null space (rank ~75 of 306): projected
out, never inverted."""
STRIDE = 2
"""Every how many samples the reference reads of each subject-0 run; its covariance needs far fewer than a run has."""
EVERY = 4
"""Every how many of subject 0's training runs the reference reads: 39 of 156 still span every book and session."""
QUANTILE_SAMPLE = 20000
REFERENCE = "sub-0_"
"""The prefix of subject 0's run names in both the training cache and the Kaggle test cache (`subj00_`)."""
UNALIGNED = (REFERENCE, "subj00_")


@dataclass(frozen=True)
class Projection:
    """Subject 0's Maxwell-filtered signal subspace: a recording read as its coordinates along the `basis` its
    reference covariance spans, in subject 0's units, the largest component first. One fixed linear map for every
    run, so lossless on any signal inside that span."""

    median: Float[Tensor, "c 1"]
    spread: Float[Tensor, "c 1"]
    basis: Float[Tensor, "k c"]

    @property
    def rank(self) -> int:
        """Components a projected recording keeps."""
        return self.basis.shape[0]

    @torch.no_grad()
    def project(self, run: Float[Tensor, "c s"], *, robust: bool = False) -> Float[Tensor, "k s"]:
        """`run`'s coordinates in float32, from subject 0's median and range; a `robust` run, already standardised
        by its own (`DeviceRecording.standardise`), is read as it stands."""
        z = run.float() if robust else (run.float() - self.median) / self.spread
        return self.basis @ z


@dataclass(frozen=True)
class Alignment:
    """Subject 0's per-channel median and interquartile range, and the square root of its sensor covariance in those
    units: what every aligned run is re-coloured to."""

    median: Float[Tensor, "c 1"]
    spread: Float[Tensor, "c 1"]
    root: Float[Tensor, "c c"]
    exempt: tuple[str, ...] = UNALIGNED
    """Prefixes of the runs left as recorded: subject 0's for a model trained on them unaligned, none for one trained
    on aligned runs."""

    @classmethod
    def of(
        cls, runs: Sequence[Float[np.ndarray, "c s"]], device: torch.device | str, exempt: tuple[str, ...] = UNALIGNED
    ) -> "Alignment":
        """Subject 0's reference from its training `runs`: the median of their medians and ranges, the mean of their
        covariances."""
        signals = [torch.from_numpy(np.asarray(run[:, ::STRIDE], dtype=np.float32)).to(device) for run in runs]
        medians, spreads = zip(*(cls.robust(signal) for signal in signals), strict=True)
        median = torch.stack(medians).median(0).values
        spread = torch.stack(spreads).median(0).values
        covariance = torch.stack([cls.covariance((s - m) / spread) for s, m in zip(signals, medians, strict=True)])
        return cls(median, spread, cls.power(covariance.mean(0), 0.5), exempt)

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

    def projection(self) -> Projection:
        """The span of `root` — subject 0's covariance above `FLOOR` of its largest eigenvalue — as a `Projection`."""
        values, vectors = torch.linalg.eigh(self.root)
        kept = values > values.max() * FLOOR
        return Projection(self.median, self.spread, vectors[:, kept].flip(1).T.float().contiguous())

    def applies(self, name: str) -> bool:
        """Whether run `name` is re-coloured: every run but the `exempt`."""
        return not name.startswith(self.exempt)

    @torch.no_grad()
    def align(self, run: Float[Tensor, "c s"]) -> None:
        """`run` in place, in subject 0's units, whitened by its own covariance and coloured by subject 0's."""
        x = run.float()
        median, _ = self.robust(x)
        z = (x - median) / self.spread
        recolour = (self.root @ self.power(self.covariance(z), -0.5)).float()
        run.copy_((recolour @ z) * self.spread + self.median)


@dataclass(frozen=True)
class Geometry:
    """The sensor geometry recordings load in: re-coloured by `alignment` when given, then kept as their coordinates
    in `projection`'s subspace when given; as recorded without either."""

    alignment: Alignment | None = None
    projection: Projection | None = None

    def rows(self, channels: int) -> int:
        """Rows a recording of `channels` sensors loads as."""
        return channels if self.projection is None else self.projection.rank

    def align(self, run: Float[Tensor, "c s"], name: str) -> None:
        """Run `name` in place, re-coloured when `alignment` applies to it."""
        if self.alignment is not None and self.alignment.applies(name):
            self.alignment.align(run)

    def project(self, run: Float[Tensor, "c s"], *, robust: bool = False) -> Float[Tensor, "r s"]:
        """`run` in `projection`'s subspace (`Projection.project`), or as it is without one."""
        return run if self.projection is None else self.projection.project(run, robust=robust)


AS_RECORDED = Geometry()

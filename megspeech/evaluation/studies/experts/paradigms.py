"""The classical BCI paradigms' features (Vourvopoulos & Perdikis, ch. 3) as the peer booster's side information
(megspeech-508): does any of them add to what the frozen base already reads from a word's window?

Each word's 1 s window, reduced to the leading spatial components of the out-of-fold windows, gives four groups:
log band power (sensorimotor, passive, affective and error-theta rhythms), 100 ms window means (the P300 and evoked
responses' features), the log-Euclidean covariance tangent (the Riemannian features), and the uncentred window level
(slow cortical potentials, which the centred model input never sees). Each group is one arm beside a base-only control.

    uv run python -m megspeech.evaluation.studies experts.paradigms --folds kf5t-f0 kf5t-f1 kf5t-f2 kf5t-f3 kf5t-f4 \\
        --base peerbase-s0
"""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.constants import RATE
from megspeech.data.raw.runs import Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.studies.experts.boosted import Design, Rows
from megspeech.evaluation.studies.experts.peer import Dump, Study

LOG = logging.getLogger(__name__)

BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 25.0))
"""Delta, theta, alpha (mu) and beta in Hz, as ch. 3 names them; beta stops at the 50 Hz input's Nyquist."""
COMPONENTS = 32
"""Spatial components the features are read from: under half the ~75-dimensional Maxwell-filtered sensor space."""
COVARIANCE_COMPONENTS = 12
"""Components the covariance reads: 12 x 12 from a window's 50 samples stays well conditioned."""
SEGMENTS = 10
"""Window means per window: 100 ms each (Eq. 3.2's non-overlapping windows)."""
SHRINK = 1e-3
"""Ridge on a window covariance, relative to its mean variance, before its matrix logarithm."""
CHUNK = 2048
"""Words whose windows are read at once."""


class Paradigm(StrEnum):
    BAND = "band"
    MEANS = "means"
    COVARIANCE = "covariance"
    LEVEL = "level"


@dataclass(frozen=True)
class Spatial:
    """The leading principal directions of some windows' sensor covariance."""

    basis: Float[Tensor, "c k"]

    @classmethod
    def fitted(cls, splits: Iterable[Split], components: int) -> "Spatial":
        total: Tensor | None = None
        for split in splits:
            for words in torch.arange(len(split.y), device=split.windows.device).split(CHUNK):
                x = split.windows(words).float()
                part = torch.einsum("kcs,kds->cd", x, x)
                total = part if total is None else total + part
        if total is None:
            raise ValueError("no windows to fit the spatial components on")
        _, vectors = torch.linalg.eigh(total)
        return cls(vectors[:, -components:].flip(-1))

    def __call__(self, x: Float[Tensor, "k c s"]) -> Float[Tensor, "k m s"]:
        return torch.einsum("kcs,cm->kms", x, self.basis)


@dataclass(frozen=True)
class Extractor:
    """Each word's paradigm features from its window reduced by `spatial`."""

    spatial: Spatial

    @staticmethod
    def band(x: Float[Tensor, "k m s"]) -> Float[Tensor, "k f"]:
        spectrum = torch.fft.rfft(x * torch.hann_window(x.shape[-1], device=x.device), dim=-1).abs().square()
        hertz = torch.fft.rfftfreq(x.shape[-1], 1 / RATE).to(x.device)
        power = [spectrum[..., (hertz >= low) & (hertz < high)].sum(-1) for low, high in BANDS]
        return torch.stack(power, dim=-1).clamp(min=torch.finfo(x.dtype).tiny).log().flatten(1)

    @staticmethod
    def means(x: Float[Tensor, "k m s"]) -> Float[Tensor, "k f"]:
        return x.unflatten(-1, (SEGMENTS, -1)).mean(-1).flatten(1)

    @staticmethod
    def covariance(x: Float[Tensor, "k m s"]) -> Float[Tensor, "k f"]:
        part = x[:, :COVARIANCE_COMPONENTS]
        covariance = part @ part.transpose(1, 2) / part.shape[-1]
        ridge = SHRINK * covariance.diagonal(dim1=1, dim2=2).mean(-1).clamp(min=torch.finfo(x.dtype).tiny)
        covariance = covariance + ridge[:, None, None] * torch.eye(part.shape[1], device=x.device)
        values, vectors = torch.linalg.eigh(covariance)
        logged = vectors @ torch.diag_embed(values.log()) @ vectors.transpose(1, 2)
        upper = torch.triu_indices(part.shape[1], part.shape[1], device=x.device)
        return logged[:, upper[0], upper[1]]

    def __call__(self, split: Split) -> dict[Paradigm, Float[Tensor, "n f"]]:
        uncentred = replace(split.windows, centred=False)
        parts: dict[Paradigm, list[Tensor]] = {paradigm: [] for paradigm in Paradigm}
        for words in torch.arange(len(split.y), device=split.windows.device).split(CHUNK):
            centred = self.spatial(split.windows(words).float())
            parts[Paradigm.BAND].append(self.band(centred))
            parts[Paradigm.MEANS].append(self.means(centred))
            parts[Paradigm.COVARIANCE].append(self.covariance(centred))
            parts[Paradigm.LEVEL].append(self.spatial(uncentred(words).float()).mean(-1))
        return {paradigm: torch.cat(chunks).cpu() for paradigm, chunks in parts.items()}


@dataclass(frozen=True)
class Features:
    """Every out-of-fold and validation word's paradigm features, in the dump's row order."""

    oof: dict[Paradigm, Float[Tensor, "n f"]]
    val: dict[Paradigm, Float[Tensor, "m f"]]

    @staticmethod
    def joined(parts: list[dict[Paradigm, Tensor]]) -> dict[Paradigm, Tensor]:
        return {paradigm: torch.cat([part[paradigm] for part in parts]) for paradigm in Paradigm}

    def designs(self, oof: Rows, val: Rows, arm: "Arm") -> tuple[Design, Design]:
        """The booster's training and validation designs: the `arm`'s features beside the base's log-probabilities, or
        alone over a uniform base when `arm.alone`."""

        def design(rows: Rows, features: dict[Paradigm, Tensor]) -> Design:
            base = torch.zeros_like(rows.base) if arm.alone else rows.base
            columns = [*(() if arm.alone else (rows.base,)), *(features[kind] for kind in arm.kinds)]
            return Design(torch.cat(columns, dim=1), base, rows.y)

        return design(oof, self.oof), design(val, self.val)

    def save(self, path: Path) -> None:
        torch.save({part: {k.value: v for k, v in getattr(self, part).items()} for part in ("oof", "val")}, path)

    @classmethod
    def load(cls, path: Path) -> "Features":
        saved = torch.load(path)
        return cls({Paradigm(k): v for k, v in saved["oof"].items()}, {Paradigm(k): v for k, v in saved["val"].items()})


@dataclass(frozen=True)
class Arm:
    """Which paradigm features a booster reads, and whether it reads them without the base: what they decode alone."""

    name: str
    kinds: tuple[Paradigm, ...]
    alone: bool = False


ARMS = (
    Arm("base-only", ()),
    *(Arm(paradigm.value, (paradigm,)) for paradigm in Paradigm),
    Arm("all", tuple(Paradigm)),
    *(Arm(f"alone-{paradigm.value}", (paradigm,), alone=True) for paradigm in Paradigm),
    Arm("alone-all", tuple(Paradigm), alone=True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", required=True, help="the cross-fit fold models the peer dump read")
    parser.add_argument("--base", required=True, help="the base the peer dump scored on the official validation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    loaded = Loader.of(Group(root, Scoring(), device), [args.base])
    domains, _ = loaded.domains(loaded.ruled(Run.discover(root.raw())))
    dump = Dump.load(Dump.path(root, args.base))
    cached = Dump.path(root, f"{args.base}-paradigms")
    if not cached.exists():
        held = [split for _, split in Dump.held(root, args.folds, device)]
        if not torch.equal(torch.cat([split.y.cpu() for split in held]), dump.oof.y):
            raise ValueError("the held-out splits are not the dump's out-of-fold rows")
        extract = Extractor(Spatial.fitted(held, COMPONENTS))
        oof, val = ([extract(split) for split in splits] for splits in (held, domains.values()))
        Features(Features.joined(oof), Features.joined(val)).save(cached)
    features = Features.load(cached)
    study = Study(dump.oof, dump.val, dump.domains, loaded.evaluator, loaded.log_prior, domains, device)
    results = [study.scored(arm.name, *features.designs(dump.oof, dump.val, arm)) for arm in ARMS]
    out = Dump.path(root, args.base).parent / f"{args.base}-paradigms.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

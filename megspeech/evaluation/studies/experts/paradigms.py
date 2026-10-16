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

    def designs(self, oof: Rows, val: Rows, kinds: tuple[Paradigm, ...]) -> tuple[Design, Design]:
        """The booster's training and validation designs: the base's log-probabilities beside the `kinds` features."""

        def design(rows: Rows, features: dict[Paradigm, Tensor]) -> Design:
            return Design(torch.cat([rows.base, *(features[kind] for kind in kinds)], dim=1), rows.base, rows.y)

        return design(oof, self.oof), design(val, self.val)


ARMS: dict[str, tuple[Paradigm, ...]] = {
    "base-only": (),
    **{paradigm.value: (paradigm,) for paradigm in Paradigm},
    "all": tuple(Paradigm),
}


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
    held = [split for _, split in Dump.held(root, args.folds, device)]
    spatial = Spatial.fitted(held, COMPONENTS)
    extract = Extractor(spatial)
    features = Features(
        Features.joined([extract(s) for s in held]), Features.joined([extract(s) for s in domains.values()])
    )
    if not torch.equal(torch.cat([s.y.cpu() for s in held]), dump.oof.y):
        raise ValueError("the held-out splits are not the dump's out-of-fold rows")
    study = Study(dump.oof, dump.val, dump.domains, loaded.evaluator, loaded.log_prior, domains, device)
    results = [study.scored(name, *features.designs(dump.oof, dump.val, kinds)) for name, kinds in ARMS.items()]
    out = Dump.path(root, args.base).parent / f"{args.base}-paradigms.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

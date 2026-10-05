"""Whitened teacher targets (megspeech-6ow.3): a teacher whose vectors share one huge mean and a few outlier directions
(GPT-2, Whisper hidden states) gives a cosine objective that those directions dominate. The targets of the training
pool centred and projected on the principal directions that carry `EXPLAINED` of their variance, each scaled to unit
variance; no validation or test run is read.

uv run python -m megspeech.objective.targets.whiten gpt2-large
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.speech import Targets

EXPLAINED = 0.9
"""Share of the targets' variance the kept directions carry; the rest is the tail whitening would inflate to noise."""
SUFFIX = "-white"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Whitening:
    """The centre and the scaled principal directions of a set of targets."""

    mean: Float[Tensor, " d"]
    basis: Float[Tensor, "d k"]

    @classmethod
    def fit(cls, runs: list[Float[np.ndarray, "n d"]]) -> "Whitening":
        """The whitening of the finite rows of `runs`, from their covariance accumulated run by run."""
        rows = [torch.from_numpy(run.astype(np.float64)) for run in runs]
        rows = [run[torch.isfinite(run).all(dim=1)] for run in rows]
        count = sum(len(run) for run in rows)
        mean = torch.stack([run.sum(0) for run in rows]).sum(0) / count
        covariance = torch.stack([(run - mean).T @ (run - mean) for run in rows]).sum(0) / count
        values, vectors = torch.linalg.eigh(covariance)
        values, vectors = values.flip(0), vectors.flip(1)
        keep = int((values.cumsum(0) / values.sum() < EXPLAINED).sum()) + 1
        return cls(mean.float(), (vectors[:, :keep] / values[:keep].sqrt()).float())

    def apply(self, vectors: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n k"]:
        """`vectors` centred and projected; absent (NaN) rows stay absent."""
        return ((torch.from_numpy(vectors.astype(np.float32)) - self.mean) @ self.basis).numpy().astype(np.float16)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("source", help="the teacher's directory under processed/speech")
        arguments = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        source = Targets.directory(root, arguments.source)
        paths = [source / f"{run.name}.npy" for run in Run.everywhere(root)]
        paths = [path for path in paths if path.exists()]
        whitening = Whitening.fit([np.load(path) for path in paths])
        destination = Path(Targets.directory(root, arguments.source + SUFFIX))
        for path in paths:
            Targets(whitening.apply(np.load(path))).save(destination / path.name)
        LOG.info("wrote %s: %d runs, %d directions", destination, len(paths), whitening.basis.shape[1])


if __name__ == "__main__":
    Whitening.main()

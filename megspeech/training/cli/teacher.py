"""How well MEG predicts each teacher (megspeech-6ow.1): a ridge from every training word's time-binned MEG window to
its teacher vector, scored on the R2 of words from runs the ridge never saw, so the layer or space an expert is
trained toward is the one the signal holds most of.

uv run python -m megspeech.training.cli.teacher whisper-medium-L8 whisper-medium-L12 gpt2-large
"""

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
from jaxtyping import Bool, Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.objective.targets.speech import Speech, Targets
from megspeech.training.config import RunConfig
from megspeech.training.loop.trainer import Trainer

BINS = 10
"""Time bins a window is averaged into: 100 ms of a 1 s window, the resolution MEG word responses are read at."""
HELD_EVERY = 5
"""One run in this many is held out of the fit, whole, so no session is in both."""
SAMPLE = 150_000
"""Most words one probe is fitted and scored on."""
CHUNK = 4096
"""Windows binned per step."""
CLIP = 5.0
"""Standard deviations a feature is clipped to: a sensor jump must not be the fit."""
RIDGES = (1e-1, 1.0, 1e1, 1e2, 1e3)
"""Penalties, as a fraction of the mean eigenvalue of the features' covariance."""
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Probe:
    """Standardised features and teacher vectors of the words that have one, the held-out runs' words marked."""

    x: Float[Tensor, "n f"]
    y: Float[Tensor, "n d"]
    held: Bool[Tensor, " n"]

    @staticmethod
    def binned(split: Split) -> Float[Tensor, "n f"]:
        """Every window of `split` averaged into `BINS` time bins per sensor, flattened."""
        everyone = torch.arange(len(split.y), device=split.y.device)
        parts = [
            nnf.adaptive_avg_pool1d(split.windows(rows).float(), BINS).flatten(1).half()
            for rows in everyone.split(CHUNK)
        ]
        return torch.cat(parts)

    @classmethod
    def of(cls, features: Float[Tensor, "n f"], positions: pd.DataFrame, speech: Speech, seed: int = 0) -> "Probe":
        """The words of `positions` (row-aligned with `features`) that `speech` has a target for, at most `SAMPLE` of
        them, both sides z-scored on the fitted words; every `HELD_EVERY`-th run is held out."""
        rows = (speech.rows >= 0).nonzero().squeeze(1).cpu().numpy()
        chosen = np.sort(np.random.default_rng(seed).permutation(rows)[:SAMPLE])
        device = features.device
        order = {run: rank for rank, run in enumerate(sorted(positions["run"].unique()))}
        held = torch.from_numpy(positions["run"].map(order).to_numpy()[chosen] % HELD_EVERY == 0).to(device)
        picked = torch.from_numpy(chosen)
        x = features[picked.to(device)].float()
        y = speech.vectors[speech.rows[picked]].to(device).float()
        fit = ~held
        x = ((x - x[fit].mean(0)) / x[fit].std(0).clamp_min(1e-6)).clamp(-CLIP, CLIP)
        y = (y - y[fit].mean(0)) / y[fit].std(0).clamp_min(1e-6)
        return cls(x, y, held)

    def r2(self, ridges: tuple[float, ...] = RIDGES) -> dict[float, float]:
        """Held-out R2 (all dimensions, unit variance) of the ridge at each penalty, one eigendecomposition for all."""
        fit = ~self.held
        x, y = self.x[fit], self.y[fit]
        values, basis = torch.linalg.eigh((x.T @ x / len(x)).double())
        values, vectors = values.float(), basis.float()
        cross = vectors.T @ (x.T @ y / len(x))
        scale = values.mean()
        held_x, held_y = self.x[self.held] @ vectors, self.y[self.held]
        total = (held_y**2).sum()
        return {
            ridge: float(1 - ((held_y - held_x @ (cross / (values + ridge * scale)[:, None])) ** 2).sum() / total)
            for ridge in ridges
        }

    @staticmethod
    def main() -> None:
        """Print each named teacher's best held-out R2 over `RIDGES`, as JSON lines."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("teachers", nargs="+", help="directories of per-word targets under processed/speech")
        parser.add_argument("--device", default="cuda")
        arguments = parser.parse_args()
        names = arguments.teachers
        corpus = Trainer.corpus(RunConfig(), DataRoot.from_config(), torch.device(arguments.device))
        features = Probe.binned(corpus.train)
        for name in names:
            speech = Speech.gather(Targets.directory(DataRoot.from_config(), name), corpus.positions)
            scores = Probe.of(features, corpus.positions, speech).r2()
            LOG.info("%s", json.dumps({"teacher": name, "best": max(scores.values()), "r2": scores}))


if __name__ == "__main__":
    Probe.main()

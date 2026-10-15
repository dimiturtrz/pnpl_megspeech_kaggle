"""The joint anchor target (megspeech-7gf.109.7): of each teacher, only the directions a trained MEG word encoder
reaches, fused into one per-word target for the stem's speech term.

Each teacher's ridge-CCA against the encoder's word embeddings is fit on official training words; its `K` strongest
target-side canonical variates, each scaled by its canonical correlation (the part a linear read of the MEG predicts),
are concatenated across teachers. A teacher with no target for a word contributes zeros there (its variates' mean),
so the t5 part still supervises the runs whose audio wav2vec2 never heard; a word no teacher covers has none.

    uv run python -m megspeech.evaluation.studies teachers.anchor --run ema-cd25-xsp10sem05-ac05x-s1
    uv run python -m megspeech.training.cli.train --origin ema-cd25-xsp10sem05-ac05x-s1 \\
        --optim.speech_source anchor-cca --optim.semantic 0 --name <stem>
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.runs import Column, Partition
from megspeech.device import Device
from megspeech.evaluation.studies.teachers.alignment import SEED, T5, Cca, Study, Teacher, Trained, Words
from megspeech.objective.targets.speech import Targets

LOG = logging.getLogger(__name__)

K = 32
"""Canonical axes kept per teacher: above the 11-24 family-significant components the alignment study (99y) found
per teacher, the correlation weighting fading the rest."""
FIT = 40000
"""Training words the canonical axes are fit on: tens of thousands over 512 + 1024 dimensions, as in the study."""
SOURCE = "anchor-cca"
TEACHERS = ("wav2vec2-large-xlsr-53", T5)


@dataclass(frozen=True)
class Reachable:
    """A teacher's `k` strongest canonical target axes, each weighted by its canonical correlation."""

    mean: Float[Tensor, " q"]
    axes: Float[Tensor, "q k"]

    @classmethod
    def fit(cls, x: Float[Tensor, "n p"], y: Float[Tensor, "n q"], k: int) -> "Reachable":
        cca = Cca.fit(x, y)
        a, b = cca.projected(x, y)
        weight = Cca.correlation(a[:, :k], b[:, :k]).clamp(min=0)
        LOG.info("canonical correlations: %s", " ".join(f"{value:.2f}" for value in weight.tolist()))
        return cls(cca.mean_y, cca.axes_y[:, :k] * weight)

    def projected(self, y: Float[Tensor, "n q"]) -> Float[Tensor, "n k"]:
        return (y.double() - self.mean) @ self.axes


@dataclass(frozen=True)
class Anchor:
    """Each word's (`run`, `position`) fused target, NaN where no teacher has one."""

    table: pd.DataFrame
    vectors: Float[np.ndarray, "n d"]

    @classmethod
    def fit(cls, words: Words, teachers: list[Teacher], k: int = K, limit: int = FIT) -> "Anchor":
        """Each teacher's `Reachable` part, fit on at most `limit` of `words`, side by side."""
        parts = []
        covered = np.zeros(len(words.table), dtype=bool)
        for teacher in teachers:
            present, targets = teacher.gather(words.table)
            x = words.vectors[torch.from_numpy(present)].float()
            chosen = torch.randperm(len(x), generator=torch.Generator().manual_seed(SEED))[:limit]
            part = np.zeros((len(words.table), k), dtype=np.float32)
            part[present] = Reachable.fit(x[chosen], targets[chosen], k).projected(targets).float().numpy()
            parts.append(part)
            covered |= present
        anchor = cls(words.table, np.where(covered[:, None], np.concatenate(parts, axis=1), np.nan))
        LOG.info("anchor: %d of %d words, %d dims", anchor.present.sum(), len(anchor.table), anchor.vectors.shape[1])
        return anchor

    @property
    def present(self) -> Bool[np.ndarray, " n"]:
        return np.isfinite(self.vectors).all(axis=1)

    def runs(self, lengths: dict[str, int]) -> dict[str, Targets]:
        """Each run's targets over its `lengths[run]` words, in `Run.words` order."""
        placed = {}
        for run, group in self.table.groupby(Column.RUN, sort=False):
            if len(group) != lengths[str(run)]:
                raise ValueError(f"{run}: {len(group)} words for {lengths[str(run)]} in the run; positions would shift")
            vectors = np.full((lengths[str(run)], self.vectors.shape[1]), np.nan, dtype=np.float16)
            vectors[group[Column.POSITION].to_numpy()] = self.vectors[group.index.to_numpy()]
            placed[str(run)] = Targets(vectors)
        return placed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="ema-cd25-xsp10sem05-ac05x-s1")
    parser.add_argument("--teachers", nargs="+", default=list(TEACHERS))
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--source", default=SOURCE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    runs = Trained.deep(root, Partition.TRAIN)[Partition.TRAIN]
    words = Trained.load(root, args.run, Device.CUDA.as_torch()).words(runs, np.iinfo(np.int64).max)
    anchor = Anchor.fit(words, [Study.teacher(root, name) for name in args.teachers], args.k)
    directory = Targets.directory(root, args.source)
    for run, targets in anchor.runs({run.name: len(run.words()) for run in runs}).items():
        targets.save(directory / f"{run}.npy")
    LOG.info("wrote %s", directory)


if __name__ == "__main__":
    main()

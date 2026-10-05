from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from megspeech.data.batching.split import Split
from megspeech.evaluation.studies.expert import Head
from megspeech.evaluation.studies.expertcheck import Agreement, Centroids
from megspeech.evaluation.studies.syntax import Embedded
from megspeech.model.auxiliary import Tap

WORDS = 40
DIM = 6


class Reading:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def __call__(self, _: Split) -> dict[Tap, np.ndarray]:
        return {Tap.WORD: self.vectors}


def test_of(tmp_path: Path) -> None:
    """A head that reproduces the targets wins every comparison; one that reads noise wins about half."""
    targets = np.random.default_rng(0).normal(size=(WORDS, DIM)).astype(np.float16)
    np.save(tmp_path / "run.npy", targets)
    part = pd.DataFrame({"run": ["run"] * WORDS, "position": np.arange(WORDS)})
    identity = Head(cast("Embedded", Reading(targets.astype(np.float32))), Tap.WORD, torch.eye(DIM), torch.zeros(DIM))
    exact = Agreement.of(identity, cast("Split", None), part, tmp_path)
    assert exact.words == WORDS
    assert exact.own > 0.99
    assert exact.wins > 0.9
    noise = np.random.default_rng(1).normal(size=(WORDS, DIM)).astype(np.float32)
    blind = Agreement.of(
        Head(cast("Embedded", Reading(noise)), Tap.WORD, torch.eye(DIM), torch.zeros(DIM)),
        cast("Split", None),
        part,
        tmp_path,
    )
    assert blind.own < exact.own


def centroids(tmp_path: Path) -> tuple[Centroids, np.ndarray, np.ndarray]:
    words = np.arange(WORDS) % 4
    targets = np.eye(DIM)[words] * 5 + np.random.default_rng(2).normal(size=(WORDS, DIM)) * 0.1
    np.save(tmp_path / "run.npy", targets.astype(np.float16))
    part = pd.DataFrame({"run": ["run"] * WORDS, "position": np.arange(WORDS), "target": words})
    return Centroids.fit(part, tmp_path), targets.astype(np.float32), words


def test_fit(tmp_path: Path) -> None:
    """Each word's centroid points along its own axis, and a shared direction shows as a large offset."""
    fitted, _, _ = centroids(tmp_path)
    assert fitted.means[:4].argmax(-1).tolist() == [0, 1, 2, 3]
    assert 0 < fitted.top <= 1


def test_balanced(tmp_path: Path) -> None:
    """Targets labelled by their own word are found among the nearest centroids; random outputs mostly are not."""
    fitted, targets, words = centroids(tmp_path)
    assert fitted.balanced(torch.from_numpy(targets), torch.from_numpy(words)) == 1.0
    noise = torch.from_numpy(np.random.default_rng(3).normal(size=targets.shape).astype(np.float32))
    assert fitted.balanced(noise, torch.from_numpy(words)) < 1.0

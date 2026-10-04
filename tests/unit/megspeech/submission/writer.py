from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.alignment import AS_RECORDED
from megspeech.data.cache.kaggle import KaggleCache, Track
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import SECONDARY, VOCAB_SIZE
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.model.input.subjects import Reading
from megspeech.submission import writer as module
from megspeech.submission.writer import KaggleWriter
from tests.unit.megspeech.evaluation.loaded import CPU, Favourite


class Kaggle:
    indices = np.arange(3)

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": self.indices,
                "subject": 0,
                "source": ["sentence", "sentence", "word"],
                "epoch": 0,
                "onset_s": 0.0,
            }
        )


class Cache:
    standardised: list[bool]
    geometry: list[object]

    @classmethod
    def of(cls, *_: object) -> "Cache":
        return cls()

    def build(self) -> None:
        pass

    def windows(self, device: torch.device, length: int, *, standardised: bool, geometry: object) -> Windows:
        Cache.standardised = [standardised]
        Cache.geometry = [geometry]
        return Windows.stacked(torch.zeros(3, CHANNELS, length, device=device))


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Kaggle test row is read as the zero-shot subject the reading names; the secondary vocabulary is uniform."""
    tracks: list[Track] = []
    written: dict[str, np.ndarray] = {}
    monkeypatch.setattr(module.Track, "kaggle", lambda track, _: tracks.append(track) or Kaggle())
    monkeypatch.setattr(KaggleCache, "of", Cache.of)
    monkeypatch.setattr(module, "write_submission", lambda path, _, p, s: written.update(p=p, s=s) or path)
    data = DataConfig(window=2.0, standardised=True)
    loaded = LoadedModel(
        DataRoot(tmp_path),
        data,
        CPU,
        Reading.UNSEEN,
        model=Favourite(),
        evaluator=Evaluator(CPU),
        log_prior=torch.zeros(VOCAB_SIZE),
    )
    assert KaggleWriter(Track.BROAD).write(loaded, 0.5, tmp_path / "s.csv") == tmp_path / "s.csv"
    assert (tracks, Cache.standardised, Cache.geometry) == ([Track.BROAD], [True], [AS_RECORDED])
    assert written["p"].shape == (3, VOCAB_SIZE)
    assert (written["p"].argmax(axis=1) == VOCAB_SIZE - 1).all()
    assert np.allclose(written["s"], 1 / len(SECONDARY.words))

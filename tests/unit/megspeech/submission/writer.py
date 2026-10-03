from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data import config as data_config
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.holdout import Track
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import SECONDARY, VOCAB_SIZE
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.model.input.subjects import Reading
from megspeech.submission import writer as module
from megspeech.submission.writer import HoldoutWriter
from tests.unit.megspeech.evaluation.loaded import CPU, Favourite


class Holdout:
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

    @classmethod
    def of(cls, *_: object) -> "Cache":
        return cls()

    def build(self) -> None:
        pass

    def windows(self, device: torch.device, length: int, *, standardised: bool) -> Windows:
        Cache.standardised = [standardised]
        return Windows.stacked(torch.zeros(3, CHANNELS, length, device=device))


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every holdout row is read as the zero-shot subject the reading names; the secondary vocabulary is uniform."""
    tracks: list[Track] = []
    written: dict[str, np.ndarray] = {}
    monkeypatch.setattr(module.Track, "holdout", lambda track, _: tracks.append(track) or Holdout())
    monkeypatch.setattr(data_config, "HoldoutCache", Cache)
    monkeypatch.setattr(module, "write_submission", lambda path, _, p, s: written.update(p=p, s=s) or path)
    data = DataConfig(window=2.0, standardised=True)
    loaded = LoadedModel(DataRoot(tmp_path), Favourite(), data, Evaluator(CPU), torch.zeros(VOCAB_SIZE), Reading.UNSEEN)
    assert HoldoutWriter(Track.BROAD).write(loaded, 0.5, tmp_path / "s.csv") == tmp_path / "s.csv"
    assert (tracks, Cache.standardised) == ([Track.BROAD], [True])
    assert written["p"].shape == (3, VOCAB_SIZE)
    assert (written["p"].argmax(axis=1) == VOCAB_SIZE - 1).all()
    assert np.allclose(written["s"], 1 / len(SECONDARY.words))

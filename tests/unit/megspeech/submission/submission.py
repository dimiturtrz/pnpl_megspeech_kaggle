import sys
from pathlib import Path
from typing import ClassVar, cast, override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.corpus import DataConfig
from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import WORDS
from megspeech.data.windows import Windows
from megspeech.evaluation.report import Evaluator
from megspeech.submission import submission as module
from megspeech.submission.submission import COMPETITION, FrequencyPrior, Submission, Submitter, main

VOCABULARY = ["the", "it’s", "a"]
ROWS = 4


class FakeHoldout:
    def __init__(self, *_: object, **__: object) -> None:
        self.indices = np.arange(ROWS)

    def __len__(self) -> int:
        return ROWS


def events(path: Path, words: list[str]) -> Path:
    rows = [{"kind": "silence", "segment": None}] + [{"kind": "word", "segment": word} for word in words]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_from_events(tmp_path: Path):
    """Words are lower-cased, stripped of punctuation, and straight apostrophes become the vocabulary's curly one;
    a heard `None` is a word, not a missing cell."""
    prior = FrequencyPrior.from_events([events(tmp_path / "a.tsv", ["The", "it's", "the.", "None"])])
    assert prior.counts == {"the": 2, "it’s": 1, "none": 1}


def test_probabilities():
    """Add-one smoothing: an unheard word keeps non-zero mass, and the row sums to one."""
    probabilities = FrequencyPrior(module.Counter({"the": 2, "it’s": 1})).probabilities(VOCABULARY)
    assert probabilities == pytest.approx([3 / 6, 2 / 6, 1 / 6])


def test_write_constant(tmp_path: Path):
    path = Submission(FakeHoldout()).write_constant(tmp_path / "s.csv", np.full(50, 0.02), np.full(50, 0.02))
    frame = pd.read_csv(path)
    assert frame.shape == (ROWS, 101)
    assert frame["index"].tolist() == list(range(ROWS))
    assert np.allclose(frame.iloc[:, 1:].to_numpy(), 0.02)


def test_write(tmp_path: Path):
    primary = np.tile(np.eye(50)[0], (ROWS, 1))
    path = Submission(FakeHoldout()).write(tmp_path / "s.csv", primary, np.full((ROWS, 50), 0.02))
    frame = pd.read_csv(path)
    assert frame.shape == (ROWS, 101)
    assert frame.iloc[:, 1].tolist() == [1.0] * ROWS


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The prior is read from every non-empty subject-0 events file; `--submit` uploads to the Deep track."""
    raw = tmp_path / "raw" / "pnpl_libribrain"
    raw.mkdir(parents=True)
    events(raw / "sub-0_ses-1_events.tsv", ["the"])
    (raw / "sub-0_ses-2_events.tsv").touch()
    uploads: list[tuple[Path, str]] = []
    monkeypatch.setattr(module, "LibriBrainCompetitionHoldout", FakeHoldout)
    monkeypatch.setattr(module, "submit_to_kaggle", lambda csv, competition, **_: uploads.append((csv, competition)))
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    out = tmp_path / "prior.csv"
    monkeypatch.setattr(sys, "argv", ["submission", "--out", str(out), "--submit", "smoke"])
    main()
    assert uploads == [(out, COMPETITION)]
    assert pd.read_csv(out)["the"].iloc[0] == pytest.approx(2 / 51, abs=1e-6)


class Holdout:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": [0, 1, 2],
                "subject": 0,
                "source": ["sentence", "sentence", "word"],
                "epoch": [0, 0, 0],
                "onset_s": [0.0, 0.3, 0.0],
            }
        )


class Cache:
    built: ClassVar[list[bool]] = []
    standardised: ClassVar[list[bool]] = []

    @classmethod
    def of(cls, *_: object) -> "Cache":
        return cls()

    def build(self) -> None:
        self.built.append(True)

    def windows(self, device: torch.device, length: int, *, standardised: bool) -> Windows:
        self.standardised.append(standardised)
        return Windows.stacked(torch.zeros(3, CHANNELS, length, device=device))


class Written:
    def __init__(self, holdout: object) -> None:
        self.holdout = holdout

    def write(self, path: Path, primary: np.ndarray, secondary: np.ndarray) -> Path:
        np.savez(path, primary=primary, secondary=secondary)
        return path


class Root:
    def __init__(self, raw: Path) -> None:
        self.path = raw

    def raw(self) -> Path:
        return self.path


class Constant(nn.Module):
    @override
    def forward(self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(*slot.shape, WORDS)
        logits[..., 0] = 1
        return logits, logits


def test_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "LibriBrainCompetitionHoldout", Holdout)
    monkeypatch.setattr(module, "Submission", Written)
    monkeypatch.setattr(module, "HoldoutCache", Cache)
    root = cast("DataRoot", Root(tmp_path))
    submitter = Submitter(Evaluator(torch.device("cpu")), DataConfig(window=2.0, standardised=True))
    path = submitter.submit(Constant(), root, torch.zeros(WORDS), 0.5, tmp_path / "s.npz")
    stored = np.load(path)
    assert stored["primary"].shape == (3, WORDS)
    assert np.allclose(stored["primary"].sum(axis=1), 1, atol=1e-5)
    assert Cache.built == [True]
    assert Cache.standardised == [True]

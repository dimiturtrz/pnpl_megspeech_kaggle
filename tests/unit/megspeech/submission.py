import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.paths import ENV_OVERRIDE
from megspeech import submission as module
from megspeech.submission import COMPETITION, FrequencyPrior, Submission, main

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
    """Words are lower-cased, stripped of punctuation, and straight apostrophes become the vocabulary's curly one."""
    prior = FrequencyPrior.from_events([events(tmp_path / "a.tsv", ["The", "it's", "the."])])
    assert prior.counts == {"the": 2, "it’s": 1}


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

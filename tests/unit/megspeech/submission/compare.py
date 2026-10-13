import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.submission.compare import TOP, TopTen, main

ROWS = 4


class Csv:
    """A submission whose rows rank the primary words by a per-row rotation of one fixed order."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def write(self, name: str, shifts: list[int]) -> Path:
        ranks = np.arange(len(PRIMARY))[::-1].astype(float)
        frame = pd.DataFrame([np.roll(ranks, shift) for shift in shifts], columns=list(PRIMARY.words))
        frame.insert(0, "index", np.arange(len(shifts))[::-1])
        frame["moses_am"] = 0.0
        path = self.tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        return path


def test_read(tmp_path: Path) -> None:
    """Rows come back in index order, each with exactly ten chosen words, the highest-scored ones."""
    top = TopTen.read(Csv(tmp_path).write("a", [0, 1, 2, 3]))
    assert top.index.tolist() == list(range(ROWS))
    assert (top.chosen.sum(axis=1) == TOP).all()
    assert top.chosen[-1, :TOP].all()


def test_moved(tmp_path: Path) -> None:
    """A row counts as moved when any one of its ten words changes; a reordering inside the ten does not."""
    csv = Csv(tmp_path)
    reference = TopTen.read(csv.write("ref", [0, 0, 0, 0]))
    assert TopTen.read(csv.write("same", [0, 0, 0, 0])).moved(reference) == 0.0
    assert TopTen.read(csv.write("one", [0, 0, 0, 1])).moved(reference) == 1 / ROWS


def test_overlap(tmp_path: Path) -> None:
    """A rotation by one keeps nine of ten words; rows that score different indices are refused."""
    csv = Csv(tmp_path)
    reference = TopTen.read(csv.write("ref", [0, 0, 0, 0]))
    assert TopTen.read(csv.write("one", [1, 1, 1, 1])).overlap(reference).tolist() == [TOP - 1] * ROWS
    with pytest.raises(ValueError, match="different rows"):
        TopTen.read(csv.write("short", [0, 0, 0])).overlap(reference)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Each candidate is logged against the reference."""
    csv = Csv(tmp_path)
    paths = [str(csv.write("ref", [0, 0, 0, 0])), str(csv.write("one", [0, 0, 0, 1]))]
    monkeypatch.setattr(sys, "argv", ["compare", *paths])
    with caplog.at_level(logging.INFO):
        main()
    assert "one: 0.250 rows moved, mean overlap 9.75/10" in caplog.text

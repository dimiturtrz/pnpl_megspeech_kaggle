import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.paths import DataRoot
from megspeech.data.cache.kaggle import ArchiveKind, Epochs, KaggleCache
from megspeech.data.cache.run_cache import WORD_TABLE, RunCache
from megspeech.data.labels.vocabulary import ABSENT, PRIMARY
from megspeech.data.raw.runs import Run
from megspeech.evaluation.studies.pool import overlap as module
from megspeech.evaluation.studies.pool.overlap import Onsets, Overlap, Survey, main
from megspeech.evaluation.text import Text

SLOTS = 9
SUBJECTS = 3


def table(run: str, shift: float = 0.0, words: tuple[str, ...] = ("The", "man", "said", "hello")) -> pd.DataFrame:
    return pd.DataFrame(
        {"run": run, "onset": shift + np.arange(len(words)) * 0.3, "word": list(words), "sentence": [0, 0, 1, 1]}
    )


def onsets(rows: list[list[float]], length: list[int] | None = None) -> Onsets:
    onset = np.full((len(rows), SLOTS), np.nan)
    for row, values in enumerate(rows):
        onset[row, : len(values)] = values
    lengths = np.array(length or [100] * len(rows))
    return Onsets(Epochs(lengths, np.nan_to_num(onset), ~np.isnan(onset)))


def test_of() -> None:
    reference = Text.of(table("a"))
    result = Overlap.of(reference, Text.of(table("b", shift=5.0, words=("The", "boy", "said", "hello"))), 4)
    assert (result.subject, result.words, result.total, result.sentences) == (4, 3, 4, 1)
    assert result.labelled == int((PRIMARY.labels(["The", "said", "hello"]) != ABSENT).sum())
    assert result.shift == pytest.approx(5.0)
    assert result.spread == pytest.approx(0.0)


def test_onset() -> None:
    onset = onsets([[0.0, 0.2]]).onset
    assert onset[0, :2].tolist() == [0.0, 0.2]
    assert np.isnan(onset[0, 2:]).all()


def test_equal() -> None:
    a = onsets([[0.0, 0.2], [0.0, 0.5, 0.9]])
    b = onsets([[0.0, 0.5004, 0.9], [0.0, 0.2, 0.4], [0.0, 0.2]])
    assert a.equal(b).tolist() == [[False, False, True], [True, False, False]]


def test_match() -> None:
    a = onsets([[0.0, 0.2], [0.0, 0.5, 0.9], [0.0, 0.7]], [100, 120, 90])
    b = onsets([[0.0, 0.5, 0.9], [0.0, 0.2], [0.0, 0.2]], [121, 98, 98])
    match = a.match(b, 1)
    assert (match.subject, match.matched, match.unique, match.length_spread, match.total) == (1, 2, 1, 1, 3)


def survey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Survey:
    root = DataRoot(tmp_path)
    names = ["sub-0_ses-11_task-Sherlock1_run-2", "sub-1_ses-11_task-Sherlock1_run-1", "sub-1_ses-1_task-TheMoth_run-1"]
    runs = [Run(tmp_path / f"{name}_meg.h5") for name in names]
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: runs))
    cache = RunCache.under(root.processed()).ready()
    pd.concat([table(names[0]), table(names[1], shift=2.0), table(names[2])]).to_csv(
        cache / WORD_TABLE, sep="\t", index=False
    )
    raw = KaggleCache.of(root).raw
    raw.mkdir(parents=True)
    for subject in range(SUBJECTS):
        rows = [[0.0, 0.1 * (subject + 1)], [0.0, 0.5, 0.9]]
        epochs = onsets(rows[:: 1 - 2 * (subject % 2)]).epochs
        np.savez(
            raw / f"subj{subject:02d}_{ArchiveKind.SENTENCE}.npz",
            sentence_n_times=epochs.length,
            word_onsets_s=np.where(epochs.mask, epochs.onset, np.nan),
            word_mask=epochs.mask,
            sfreq=np.float64(250),
        )
    monkeypatch.setattr(module, "N_SUBJECTS", SUBJECTS)
    return Survey(root)


def test_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (row,) = survey(tmp_path, monkeypatch).text()
    assert (row.subject, row.words, row.total, row.shift) == (1, 4, 4, pytest.approx(2.0))


def test_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matches = survey(tmp_path, monkeypatch).kaggle()
    assert [(m.subject, m.matched, m.unique) for m in matches] == [(0, 2, 2), (1, 1, 1), (2, 1, 1)]


def test_report_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=module.__name__):
        survey(tmp_path, monkeypatch).report_text()
    assert "text: 1 runs, 4 of 4 words matched" in caplog.text


def test_report_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=module.__name__):
        survey(tmp_path, monkeypatch).report_kaggle()
    assert "Kaggle test: 2 of 4 subject-0 sentences matched one-to-one across 2 other subjects" in caplog.text


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    survey(tmp_path, monkeypatch)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["overlap"])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        main()
    assert "text: 1 runs, 4 of 4 words matched" in caplog.text
    assert "Kaggle test: 2 of 4 subject-0 sentences matched one-to-one across 2 other subjects" in caplog.text

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.paths import DataRoot
from megspeech.data.cache.run_cache import WORD_TABLE
from megspeech.data.config import DataConfig, Val
from megspeech.data.constants import CHANNELS
from megspeech.data.raw.runs import Run
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.rulers import heldout as module
from megspeech.evaluation.rulers.heldout import HeldOutRuler
from megspeech.model.input.subjects import Reading
from tests.unit.megspeech.evaluation.loaded import SAMPLES, SPOKEN, loaded

NAMES = (
    "sub-124_ses-0_task-MASC1_run-1",
    "sub-124_ses-1_task-MASC2_run-1",
    "sub-125_ses-0_task-MASC1_run-1",
    "sub-1_ses-11_task-Sherlock1_run-1",
)


def runs(tmp_path: Path) -> list[Run]:
    cache = DataConfig(val=Val.OFFICIAL).run_cache(DataRoot(tmp_path)).ready()
    rows = []
    for name in NAMES:
        np.save(cache / f"{name}.npy", np.ones((CHANNELS, SAMPLES), dtype=np.float16))
        onset = 0.5 + np.arange(len(SPOKEN)) * 0.5
        rows.append(pd.DataFrame({"run": name, "onset": onset, "word": SPOKEN, "sentence": [0, 0, 0, 1, 1, 1]}))
    pd.concat(rows).assign(end=5.0).to_csv(cache / WORD_TABLE, sep="\t", index=False)
    return [Run(tmp_path / f"{name}_meg.h5") for name in NAMES]


def test_runs(tmp_path: Path) -> None:
    """Each held subject's MASC runs alone; a subject the model trained on is refused."""
    every = runs(tmp_path)
    found = HeldOutRuler((124, 125, 126)).runs(loaded(tmp_path), every)
    assert {subject: [r.name for r in rs] for subject, rs in found.items()} == {
        124: list(NAMES[:2]),
        125: [NAMES[2]],
        126: [],
    }
    tool = loaded(tmp_path)
    trained = replace(tool, data=replace(tool.data, subjects=(0, 124)))
    with pytest.raises(ValueError, match=r"\[124\]"):
        HeldOutRuler((124, 125)).runs(trained, every)


def test_subject(tmp_path: Path) -> None:
    """Every labelled word of all the subject's runs counts; another subject's reversed ranking hits none."""
    every = runs(tmp_path)
    score = HeldOutRuler((124,)).subject(loaded(tmp_path), every[:2])
    assert (score.n, score.bacc10) == (2 * len(SPOKEN), pytest.approx(0.0))


def test_select(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The macro over the held subjects with runs, each logged."""
    with caplog.at_level(logging.INFO, logger=module.__name__):
        macro = HeldOutRuler((124, 125, 126)).select(loaded(tmp_path, Reading.UNSEEN), runs(tmp_path))
    assert macro.n == 3 * len(SPOKEN)
    assert f"subject 124  runs 2  bacc10 0.0000  n {2 * len(SPOKEN)}" in caplog.text
    assert "over 2 subjects" in caplog.text


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The CLI reads every model unseen by default and scores the named subjects."""
    seen: list[Reading] = []

    def of(root: DataRoot, names: list[str], scoring: module.Scoring, device: object) -> ExportedModel:
        seen.append(scoring.reading)
        return loaded(tmp_path, scoring.reading)

    monkeypatch.setattr(module.Loader, "of", staticmethod(of))
    monkeypatch.setattr(module.DataRoot, "from_config", staticmethod(lambda: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "everywhere", staticmethod(lambda root: runs(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["heldout", "m", "--subjects", "124", "125"])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        module.main()
    assert seen == [Reading.UNSEEN]
    assert "held-out ruler" in caplog.text

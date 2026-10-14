import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.paths import DataRoot
from megspeech.data.cache.run_cache import WORD_TABLE
from megspeech.data.config import METRICS, DataConfig, Val
from megspeech.data.constants import CHANNELS
from megspeech.data.raw.runs import Run
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.rulers import heldout as module
from megspeech.evaluation.rulers.heldout import HeldOutRuler, Lineage
from megspeech.evaluation.scoring.score import Head
from megspeech.model.input.subjects import Reading
from tests.unit.megspeech.evaluation.loaded.model import SAMPLES, SPOKEN, loaded

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


def record(tmp_path: Path, name: str, subjects: list[int], stem: str = "") -> None:
    directory = DataRoot(tmp_path).models() / name
    directory.mkdir(parents=True)
    config = {"data": {"subjects": subjects}, "optim": {"stem": stem, "warm": ""}}
    (directory / METRICS).write_text(json.dumps({"config": config}))


def test_subjects(tmp_path: Path) -> None:
    """A run's subjects are the union over its stem chain; a cyclic record terminates."""
    record(tmp_path, "root", [101, 124], stem="child")
    record(tmp_path, "child", [0, 1], stem="root")
    assert Lineage(DataRoot(tmp_path).models()).subjects("child") == {0, 1, 101, 124}


def test_runs_lineage(tmp_path: Path) -> None:
    """A clean lineage passes; one whose stem trained on a held subject is refused though the model itself did not."""
    every = runs(tmp_path)
    record(tmp_path, "clean-stem", [0, 1])
    record(tmp_path, "clean", [2], stem="clean-stem")
    record(tmp_path, "dirty-stem", [0, 124])
    record(tmp_path, "dirty", [2], stem="dirty-stem")
    tool = loaded(tmp_path)
    ruler = HeldOutRuler((124, 125))
    assert set(ruler.runs(replace(tool, names=("clean",)), every)) == {124, 125}
    with pytest.raises(ValueError, match=r"\[124\]"):
        ruler.runs(replace(tool, names=("dirty",)), every)


def test_subject(tmp_path: Path) -> None:
    """Every labelled word of all the subject's runs counts; another subject's reversed ranking hits none."""
    every = runs(tmp_path)
    score = HeldOutRuler((124,)).subject(loaded(tmp_path), every[:2])
    assert (score.n, score.bacc10) == (2 * len(SPOKEN), pytest.approx(0.0))


def test_subject_head(tmp_path: Path) -> None:
    """The word head is scored through the same path as the context head and yields a score over the same words."""
    every = runs(tmp_path)
    score = HeldOutRuler((124,), Head.WORD).subject(loaded(tmp_path), every[:2])
    assert score.n == 2 * len(SPOKEN)


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

    def of(group: Group, names: list[str]) -> ExportedModel:
        seen.append(group.scoring.reading)
        return loaded(tmp_path, group.scoring.reading)

    monkeypatch.setattr(module.Loader, "of", staticmethod(of))
    monkeypatch.setattr(module.DataRoot, "from_config", staticmethod(lambda: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "everywhere", staticmethod(lambda root: runs(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["heldout", "m", "--subjects", "124", "125"])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        module.main()
    assert seen == [Reading.UNSEEN]
    assert "held-out ruler" in caplog.text

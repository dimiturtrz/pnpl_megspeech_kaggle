import logging
from dataclasses import replace
from pathlib import Path

import pytest

from megspeech.evaluation.rulers import broad as module
from megspeech.evaluation.rulers.broad import BroadRuler
from tests.unit.megspeech.evaluation.loaded import SPOKEN, loaded, runs


def test_select(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Each pooled subject counts on its own unseen spans, or not at all below `min_novel`."""
    with caplog.at_level(logging.INFO, logger=module.__name__):
        macro = BroadRuler.select(loaded(tmp_path), runs(tmp_path))
    assert (macro.n, macro.bacc10) == (len(SPOKEN), pytest.approx(0.0))
    assert f"subject  1  bacc10 0.0000  n {len(SPOKEN)}" in caplog.text
    assert f"over 1 subjects, {len(SPOKEN)} words" in caplog.text
    tool = loaded(tmp_path)
    strict = replace(tool, evaluator=replace(tool.evaluator, min_novel=len(SPOKEN) + 1))
    assert BroadRuler.select(strict, runs(tmp_path)).n == 0


def test_graded(tmp_path: Path) -> None:
    """Each counted pooled subject's unseen-span hits, keyed by subject; none below `min_novel`."""
    graded = BroadRuler.graded(loaded(tmp_path), runs(tmp_path), 0.5)
    assert {name: len(labels) for name, (_, labels) in graded.domains.items()} == {"1": len(SPOKEN)}
    tool = loaded(tmp_path)
    strict = replace(tool, evaluator=replace(tool.evaluator, min_novel=len(SPOKEN) + 1))
    assert BroadRuler.graded(strict, runs(tmp_path), 0.5).domains == {}


def test_scored(tmp_path: Path) -> None:
    """Session 11 by default and under `adapt=12`; session 12 once the model adapts on 11."""
    every = runs(tmp_path)
    tool = loaded(tmp_path)
    keys = {
        adapt: [r.key[1] for r in BroadRuler.scored(replace(tool, data=replace(tool.data, adapt=adapt)), every)]
        for adapt in (0, 11, 12)
    }
    assert keys == {0: [11], 11: [12], 12: [11]}


def test_select_held_out(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An adapted model's mean over the subjects outside `data.subjects` is logged; a default model's is not."""
    tool = loaded(tmp_path)
    adapted = replace(tool, data=replace(tool.data, adapt=11, subjects=(0, 2)))
    with caplog.at_level(logging.INFO, logger=module.__name__):
        BroadRuler.select(adapted, runs(tmp_path))
    assert "held-out subjects 1  bacc10 0.0000" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=module.__name__):
        BroadRuler.select(tool, runs(tmp_path))
    assert "held-out" not in caplog.text

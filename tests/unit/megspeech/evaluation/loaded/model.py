from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.batching.split import SpanInput
from megspeech.data.cache.run_cache import WORD_TABLE as TABLE
from megspeech.data.config import DataConfig, Val
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.text import RULER
from megspeech.model.input.subjects import Reading

CPU = torch.device("cpu")
SAMPLES = 500
SPOKEN = PRIMARY.words[:6]
NAMES = (
    "sub-0_ses-11_task-Sherlock1_run-2",
    "sub-0_ses-29_task-TheMoth_run-1",
    "sub-1_ses-11_task-Sherlock1_run-1",
    "sub-1_ses-12_task-Sherlock1_run-1",
    "sub-0_ses-12_task-Sherlock1_run-2",
)
POOLED, TRAINED, TESTED = 2, 3, 4


class Favourite(nn.Module):
    """Ranks the vocabulary by index for subject 0 and in reverse for every other subject."""

    @override
    def forward(self, spans: SpanInput) -> tuple[torch.Tensor, torch.Tensor]:
        ranks = -torch.arange(VOCAB_SIZE, dtype=torch.float32)
        rows = torch.where(spans.subject[:, None] == 0, ranks, ranks.flip(0))
        logits = nn.functional.pad(rows, (0, 0, 0, 1))[spans.slot]
        return logits, logits


def runs(tmp_path: Path) -> list[Run]:
    cache = DataConfig(val=Val.OFFICIAL).run_cache(DataRoot(tmp_path)).ready()
    rows = []
    for name in NAMES:
        np.save(cache / f"{name}.npy", np.ones((CHANNELS, SAMPLES), dtype=np.float16))
        onset = 0.5 + np.arange(len(SPOKEN)) * 0.5
        rows.append(pd.DataFrame({"run": name, "onset": onset, "word": SPOKEN, "sentence": [0, 0, 0, 1, 1, 1]}))
    pd.concat(rows).assign(end=5.0).to_csv(cache / TABLE, sep="\t", index=False)
    return [Run(tmp_path / f"{name}_meg.h5") for name in NAMES]


def loaded(tmp_path: Path, reading: Reading = Reading.OWN) -> ExportedModel:
    evaluator = Evaluator(CPU, min_novel=1)
    return ExportedModel(
        DataRoot(tmp_path),
        DataConfig(val=Val.OFFICIAL),
        CPU,
        reading,
        model=Favourite(),
        evaluator=evaluator,
        log_prior=torch.zeros(VOCAB_SIZE),
    )


def test_calibration(tmp_path: Path) -> None:
    """The model's own prior, at the given tau, balanced as its evaluator balances."""
    model = loaded(tmp_path)
    calibration = model.calibration(0.5)
    assert (calibration.log_prior, calibration.tau) == (model.log_prior, 0.5)
    assert calibration.balance == model.evaluator.balance


def test_context(tmp_path: Path) -> None:
    domains, _ = loaded(tmp_path).domains({RULER: runs(tmp_path)[:1]})
    assert loaded(tmp_path).context(domains[RULER]).argmax(dim=1).tolist() == [0] * len(SPOKEN)


def test_labelled(tmp_path: Path) -> None:
    """Another subject's reversed ranking hits none of its words."""
    (_, split), *_ = loaded(tmp_path).splits(runs(tmp_path)[TRAINED:]).values()
    score = loaded(tmp_path).labelled(split)
    assert (score.n, score.bacc10) == (len(SPOKEN), pytest.approx(0.0))


def test_unseen(tmp_path: Path) -> None:
    """Every word of the pooled run is novel to a model that saw no orders."""
    score = loaded(tmp_path).unseen(runs(tmp_path)[POOLED])
    assert (score.n, score.bacc10) == (len(SPOKEN), pytest.approx(0.0))


def test_report(tmp_path: Path) -> None:
    report = loaded(tmp_path).report(runs(tmp_path))
    assert RULER in report.domains
    assert report.macro.n == sum(r.novel.n for r in report.domains.values())


def test_report_tested(tmp_path: Path) -> None:
    """The test partition scores subject 0's official test run alone, not the validation domains."""
    report = replace(loaded(tmp_path), partition=Partition.TEST).report(runs(tmp_path))
    assert list(report.domains) == ["Sherlock1"]
    assert report.macro.n == report.domains["Sherlock1"].novel.n > 0


def test_graded(tmp_path: Path) -> None:
    """The hits re-score to the report's own macro at its selected adjustment."""
    report = loaded(tmp_path).report(runs(tmp_path))
    graded = loaded(tmp_path).graded(runs(tmp_path), report.tau)
    assert Ruler().point(graded) == pytest.approx(report.macro.bacc10)
    assert sum(len(labels) for _, labels in graded.domains.values()) == report.macro.n


def test_domain_logits(tmp_path: Path) -> None:
    domains, logits = loaded(tmp_path).domain_logits(runs(tmp_path))
    assert RULER in domains
    assert logits[RULER][Head.CONTEXT].shape == (len(domains[RULER].y), VOCAB_SIZE)


def test_adapted_on(tmp_path: Path) -> None:
    tool = loaded(tmp_path)
    assert tool.adapted_on == 0
    assert replace(tool, data=replace(tool.data, subjects=(0, 1), adapt=12)).adapted_on == 12


def test_run_context(tmp_path: Path) -> None:
    split, context = loaded(tmp_path).run_context(runs(tmp_path)[POOLED])
    assert len(split.y) == len(context) == len(SPOKEN)
    assert context.argmax(dim=1).tolist() == [VOCAB_SIZE - 1] * len(SPOKEN)


def test_score(tmp_path: Path) -> None:
    """Subject 0's ranking puts every word in the top ten of the context scored against its own labels."""
    domains, _ = loaded(tmp_path).domains({RULER: runs(tmp_path)[:1]})
    split = domains[RULER]
    score = loaded(tmp_path).score(loaded(tmp_path).context(split), split.y)
    assert (score.n, score.bacc10) == (len(SPOKEN), pytest.approx(1.0))


def test_broad(tmp_path: Path) -> None:
    """Session 11 by default and under `adapt=12`; session 12 once the model adapts on 11."""
    every = runs(tmp_path)
    tool = loaded(tmp_path)
    keys = {
        adapt: [r.key[1] for r in replace(tool, data=replace(tool.data, adapt=adapt)).broad(every)]
        for adapt in (0, 11, 12)
    }
    assert keys == {0: [11], 11: [12], 12: [11]}

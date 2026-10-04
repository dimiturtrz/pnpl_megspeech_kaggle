import json
from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.cache.run_cache import WORD_TABLE as TABLE
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation import loaded as module
from megspeech.evaluation import reader as reader_module
from megspeech.evaluation.loaded import LoadedModel, Scoring
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.evaluator import Evaluator
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
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ranks = -torch.arange(VOCAB_SIZE, dtype=torch.float32)
        rows = torch.where(subject[:, None] == 0, ranks, ranks.flip(0))
        logits = nn.functional.pad(rows, (0, 0, 0, 1))[slot]
        return logits, logits


def runs(tmp_path: Path) -> list[Run]:
    cache = DataConfig().run_cache(DataRoot(tmp_path)).ready()
    rows = []
    for name in NAMES:
        np.save(cache / f"{name}.npy", np.ones((CHANNELS, SAMPLES), dtype=np.float16))
        onset = 0.5 + np.arange(len(SPOKEN)) * 0.5
        rows.append(pd.DataFrame({"run": name, "onset": onset, "word": SPOKEN, "sentence": [0, 0, 0, 1, 1, 1]}))
    pd.concat(rows).assign(end=5.0).to_csv(cache / TABLE, sep="\t", index=False)
    return [Run(tmp_path / f"{name}_meg.h5") for name in NAMES]


def loaded(tmp_path: Path, reading: Reading = Reading.OWN) -> LoadedModel:
    evaluator = Evaluator(CPU, min_novel=1)
    return LoadedModel(
        DataRoot(tmp_path),
        DataConfig(),
        CPU,
        reading,
        model=Favourite(),
        evaluator=evaluator,
        log_prior=torch.zeros(VOCAB_SIZE),
    )


def members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selections: dict[str, int]) -> None:
    """Trained runs named by `selections`, each recording its `min_novel`, behind a stubbed word table and model."""
    heard = pd.DataFrame(
        {"run": "r", "sentence": 0, "onset": [0.0, 0.1], "target": [0, 1], "token": [7, 8], "weight": 1.0}
    )
    monkeypatch.setattr(reader_module.Corpus, "heard", classmethod(lambda *_: heard))
    monkeypatch.setattr(reader_module.Run, "discover", classmethod(lambda *_: []))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: Favourite()))
    for name, min_novel in selections.items():
        directory = DataRoot(tmp_path).models() / name
        directory.mkdir(parents=True)
        config: dict[str, dict[str, int]] = {"data": {}, "selection": {"min_novel": min_novel}}
        (directory / METRICS).write_text(json.dumps({"config": config}))


def test_loaded_model_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Members are scored by the prior training drew and the selection that counted them, or not at all."""
    members(tmp_path, monkeypatch, {"a": 5, "b": 5, "c": 7})
    (DataRoot(tmp_path).models() / "ab").mkdir()
    (DataRoot(tmp_path).models() / "ab" / METRICS).write_text(json.dumps({"members": ["a", "b"]}))
    scoring = Scoring(balance=Sinkhorn(2), shifts=(0, 1), reading=Reading.UNSEEN)
    built = LoadedModel.of(DataRoot(tmp_path), ["ab"], scoring, CPU)
    expected = torch.tensor([1.0, 1.0] + [0.0] * (VOCAB_SIZE - 2)) + 1
    assert torch.allclose(built.log_prior, (expected / expected.sum()).log())
    assert built.evaluator.seen == frozenset({(7, 8)})
    assert (built.evaluator.min_novel, built.evaluator.balance, built.evaluator.shifts) == (5, Sinkhorn(2), (0, 1))
    assert (built.names, built.reading) == (("a", "b"), Reading.UNSEEN)
    with pytest.raises(ValueError, match="an ensemble needs one"):
        LoadedModel.of(DataRoot(tmp_path), ["b", "c"], scoring, CPU)


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


def test_probabilities(tmp_path: Path) -> None:
    domains, _ = loaded(tmp_path).domains({RULER: runs(tmp_path)[:1]})
    probabilities = loaded(tmp_path).probabilities(domains[RULER], 0.5)
    assert probabilities.shape == (len(SPOKEN), VOCAB_SIZE)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert (probabilities.argmax(axis=1) == 0).all()


class Contrary(Favourite):
    """`Favourite`'s context head, with a word head that ranks the vocabulary in reverse."""

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, _ = super().forward(x, onset, slot, subject)
        return context, context.flip(-1)


def test_probabilities_lone(tmp_path: Path) -> None:
    """A word alone in its span reads the word head; a word with context reads the context head."""
    domains, _ = loaded(tmp_path).domains({RULER: runs(tmp_path)[:1]})
    split = domains[RULER]
    lone = torch.full((3, split.index.shape[1]), PAD)
    lone[:, 0] = split.index[1, :3]
    split = replace(split, index=torch.cat([split.index[:1], lone]), onset=torch.zeros(4, split.index.shape[1]))
    probabilities = replace(loaded(tmp_path), model=Contrary()).probabilities(split, 0.5)
    assert probabilities.argmax(axis=1).tolist() == [0, 0, 0, VOCAB_SIZE - 1, VOCAB_SIZE - 1, VOCAB_SIZE - 1]

import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.batching.split import FEATURES
from megspeech.data.cache.run_cache import WORD_TABLE as TABLE
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.evaluation import pooling as module
from megspeech.evaluation.balance import Sinkhorn
from megspeech.evaluation.overlap import RULER
from megspeech.evaluation.pooling import CHANCE, Equal, Pool, Pooling, Reliable, Scoring, main
from megspeech.evaluation.report import Evaluator

CPU = torch.device("cpu")
SAMPLES = 500
SPOKEN = PRIMARY.words[:6]
NAMES = (
    "sub-0_ses-11_task-Sherlock1_run-2",
    "sub-0_ses-29_task-TheMoth_run-1",
    "sub-1_ses-11_task-Sherlock1_run-1",
    "sub-1_ses-12_task-Sherlock1_run-1",
)


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
    root = DataRoot(tmp_path)
    cache = DataConfig().run_cache(root).ready()
    rows = []
    for name in NAMES:
        np.save(cache / f"{name}.npy", np.ones((CHANNELS, SAMPLES), dtype=np.float16))
        onset = 0.5 + np.arange(len(SPOKEN)) * 0.5
        rows.append(pd.DataFrame({"run": name, "onset": onset, "word": SPOKEN, "sentence": [0, 0, 0, 1, 1, 1]}))
    pd.concat(rows).assign(end=5.0).to_csv(cache / TABLE, sep="\t", index=False)
    return [Run(tmp_path / f"{name}_meg.h5") for name in NAMES]


def pooling(tmp_path: Path) -> Pooling:
    evaluator = Evaluator(CPU, min_novel=1)
    return Pooling(DataRoot(tmp_path), Favourite(), DataConfig(), evaluator, torch.zeros(VOCAB_SIZE))


def test_equal_weights() -> None:
    assert Equal().weights(torch.tensor([0.1, 0.9])).tolist() == [1.0, 1.0]


def test_reliable_weights() -> None:
    weights = Reliable().weights(torch.tensor([CHANCE - 0.1, CHANCE + 0.1, CHANCE + 0.3]))
    assert weights.tolist() == pytest.approx([0.0, 0.75, 2.25])
    assert Reliable().weights(torch.full((2,), CHANCE)).tolist() == [0.0, 0.0]


def test_combine() -> None:
    generator = torch.Generator().manual_seed(0)
    reference, others = (
        torch.randn(3, VOCAB_SIZE, generator=generator),
        torch.randn(2, 3, VOCAB_SIZE, generator=generator),
    )
    present = torch.tensor([[True, True, False], [True, False, False]])
    reliability = torch.ones(2)
    alone = Pool(math.inf, Equal()).combine(reference, others, present, reliability)
    assert torch.allclose(alone, reference.log_softmax(-1))
    pooled = Pool(2.0, Equal()).combine(reference, others, present, reliability)
    logs = reference.log_softmax(-1), others.log_softmax(-1)
    assert torch.allclose(pooled[0], (2 * logs[0][0] + logs[1][0, 0] + logs[1][1, 0]) / 4)
    assert torch.allclose(pooled[1], (2 * logs[0][1] + logs[1][0, 1]) / 3)
    assert torch.allclose(pooled[2], logs[0][2])


def members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selections: dict[str, int]) -> list[Path]:
    """Trained runs named by `selections`, each recording its `min_novel`, behind a stubbed word table and model."""
    heard = pd.DataFrame(
        {"run": "r", "sentence": 0, "onset": [0.0, 0.1], "target": [0, 1], "token": [7, 8], "weight": 1.0}
    )
    monkeypatch.setattr(module.Corpus, "heard", classmethod(lambda *_: heard))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: []))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: Favourite()))
    directories = [DataRoot(tmp_path).models() / name for name in selections]
    for directory, min_novel in zip(directories, selections.values(), strict=True):
        directory.mkdir(parents=True)
        config: dict[str, dict[str, int]] = {"data": {}, "selection": {"min_novel": min_novel}}
        (directory / METRICS).write_text(json.dumps({"config": config}))
    return directories


def test_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    members(tmp_path, monkeypatch, {"m": 3})
    built = Pooling.of(DataRoot(tmp_path), "m", DataConfig().grouping, CPU)
    assert built.evaluator.seen == frozenset({(7, 8)})
    assert built.evaluator.min_novel == 3
    assert built.log_prior.shape == (VOCAB_SIZE,)


def test_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Members are scored by the prior training drew and the selection that counted them, or not at all."""
    directories = members(tmp_path, monkeypatch, {"a": 5, "b": 5, "c": 7})
    scoring = Scoring(DataConfig().grouping, CPU, balance=Sinkhorn(2), shifts=(0, 1))
    built = Pooling.loaded(DataRoot(tmp_path), directories[:2], scoring)
    expected = torch.tensor([1.0, 1.0] + [0.0] * (VOCAB_SIZE - 2)) + 1
    assert torch.allclose(built.log_prior, (expected / expected.sum()).log())
    assert (built.evaluator.min_novel, built.evaluator.balance, built.evaluator.shifts) == (5, Sinkhorn(2), (0, 1))
    with pytest.raises(ValueError, match="an ensemble needs one"):
        Pooling.loaded(DataRoot(tmp_path), directories[1:], scoring)


def test_features(tmp_path: Path) -> None:
    assert pooling(tmp_path).features.directory is None
    frozen = replace(pooling(tmp_path), data=DataConfig(features="frozen"))
    assert frozen.features.directory == DataRoot(tmp_path).kind("models") / FEATURES / "frozen"


def test_split(tmp_path: Path) -> None:
    """A run reads its MEG windows, or fails loudly where a features model has no features saved for it."""
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    source, words = tool.load(every[2:3])
    assert len(tool.split(source, words, every[2]).y) == len(SPOKEN)
    frozen = replace(tool, data=DataConfig(features="frozen"))
    with pytest.raises(FileNotFoundError):
        frozen.split(source, words, every[2])


def test_members(tmp_path: Path) -> None:
    assert Pooling.members(tmp_path, "run") == [tmp_path / "run"]
    (tmp_path / "trained").mkdir()
    (tmp_path / "trained" / "metrics.json").write_text(json.dumps({"best": {}}))
    assert Pooling.members(tmp_path, "trained") == [tmp_path / "trained"]
    (tmp_path / "ens").mkdir()
    (tmp_path / "ens" / "metrics.json").write_text(json.dumps({"members": ["a", "b"]}))
    assert Pooling.members(tmp_path, "ens") == [tmp_path / "a", tmp_path / "b"]


def test_load(tmp_path: Path) -> None:
    every = runs(tmp_path)
    source, words = pooling(tmp_path).load(every[2:])
    assert source.names == NAMES[2:]
    assert words["position"].tolist() == [0, 1, 2, 0, 1, 2] * 2


def test_domains(tmp_path: Path) -> None:
    every = runs(tmp_path)
    domains, reference = pooling(tmp_path).domains({RULER: every[:1], "TheMoth": every[1:2]})
    assert set(domains) == {RULER, "TheMoth"}
    assert (reference["run"] == NAMES[0]).all()
    assert len(reference) == len(domains[RULER].y)
    unruled, empty = pooling(tmp_path).domains({"TheMoth": every[1:2]})
    assert set(unruled) == {"TheMoth"}
    assert empty.empty
    frozen = replace(pooling(tmp_path), data=DataConfig(features="frozen"))
    with pytest.raises(FileNotFoundError):
        frozen.domains({"TheMoth": every[1:2]})


def test_context(tmp_path: Path) -> None:
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    domains, _ = tool.domains({RULER: every[:1]})
    assert tool.context(domains[RULER]).argmax(dim=1).tolist() == [0] * len(SPOKEN)


def test_echo(tmp_path: Path) -> None:
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    _, reference = tool.domains({RULER: every[:1]})
    logits, present, reliability = tool.echo(reference.iloc[:4], every[2], every[3])
    assert present.tolist() == [True] * 4
    assert logits.argmax(dim=1).tolist() == [VOCAB_SIZE - 1] * 4
    assert reliability == pytest.approx(0.0)


def test_echoes(tmp_path: Path) -> None:
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    _, reference = tool.domains({RULER: every[:1]})
    echoes = tool.echoes(reference, every)
    assert echoes.logits.shape == (1, len(SPOKEN), VOCAB_SIZE)
    assert echoes.present.all()
    assert echoes.reliability.shape == (1,)


def test_macro(tmp_path: Path) -> None:
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    domains, _ = tool.domains({RULER: every[:1], "TheMoth": every[1:2]})
    context = {name: tool.context(split) for name, split in domains.items()}
    macro = tool.macro(domains, context)
    assert macro.n == 2 * len(SPOKEN)
    hits = sum(label < 10 for label in range(len(SPOKEN))) / len(SPOKEN)
    assert macro.bacc10 == pytest.approx(hits)


def test_matched(tmp_path: Path) -> None:
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    domains, _ = tool.domains({RULER: every[:1]})
    context = tool.context(domains[RULER])
    present = torch.tensor([True, True, False, False, False, False])
    assert tool.matched(domains[RULER], context, present).n == 2


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    every = runs(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: DataRoot(tmp_path)))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: every))
    monkeypatch.setattr(module.Pooling, "of", classmethod(lambda *_: pooling(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["pooling", "m"])
    with caplog.at_level(logging.INFO, logger=module.__name__):
        main()
    assert f"ruler words {len(SPOKEN)}, heard by another subject {len(SPOKEN)}" in caplog.text
    assert caplog.text.count("macro") == 2 * len(module.WEIGHTS)

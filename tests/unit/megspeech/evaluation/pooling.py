import json
import logging
import math
import sys
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.continuous import WORDS as TABLE
from megspeech.data.corpus import DataConfig
from megspeech.data.runs import CHANNELS, Run
from megspeech.data.vocabulary import PRIMARY, WORDS
from megspeech.evaluation import pooling as module
from megspeech.evaluation.overlap import RULER
from megspeech.evaluation.pooling import CHANCE, Equal, Pool, Pooling, Reliable, main
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
        ranks = -torch.arange(WORDS, dtype=torch.float32)
        rows = torch.where(subject[:, None] == 0, ranks, ranks.flip(0))
        logits = nn.functional.pad(rows, (0, 0, 0, 1))[slot]
        return logits, logits


def runs(tmp_path: Path) -> list[Run]:
    root = DataRoot(tmp_path)
    cache = DataConfig().decimator(root).ready()
    rows = []
    for name in NAMES:
        np.save(cache / f"{name}.npy", np.ones((CHANNELS, SAMPLES), dtype=np.float16))
        onset = 0.5 + np.arange(len(SPOKEN)) * 0.5
        rows.append(pd.DataFrame({"run": name, "onset": onset, "word": SPOKEN, "sentence": [0, 0, 0, 1, 1, 1]}))
    pd.concat(rows).assign(end=5.0).to_csv(cache / TABLE, sep="\t", index=False)
    return [Run(tmp_path / f"{name}_meg.h5") for name in NAMES]


def pooling(tmp_path: Path) -> Pooling:
    evaluator = Evaluator(CPU, min_novel=1)
    return Pooling(DataRoot(tmp_path), Favourite(), DataConfig(), evaluator, torch.zeros(WORDS))


def test_equal_weights() -> None:
    assert Equal().weights(torch.tensor([0.1, 0.9])).tolist() == [1.0, 1.0]


def test_reliable_weights() -> None:
    weights = Reliable().weights(torch.tensor([CHANCE - 0.1, CHANCE + 0.1, CHANCE + 0.3]))
    assert weights.tolist() == pytest.approx([0.0, 0.75, 2.25])
    assert Reliable().weights(torch.full((2,), CHANCE)).tolist() == [0.0, 0.0]


def test_combine() -> None:
    generator = torch.Generator().manual_seed(0)
    reference, others = torch.randn(3, WORDS, generator=generator), torch.randn(2, 3, WORDS, generator=generator)
    present = torch.tensor([[True, True, False], [True, False, False]])
    reliability = torch.ones(2)
    alone = Pool(math.inf, Equal()).combine(reference, others, present, reliability)
    assert torch.allclose(alone, reference.log_softmax(-1))
    pooled = Pool(2.0, Equal()).combine(reference, others, present, reliability)
    logs = reference.log_softmax(-1), others.log_softmax(-1)
    assert torch.allclose(pooled[0], (2 * logs[0][0] + logs[1][0, 0] + logs[1][1, 0]) / 4)
    assert torch.allclose(pooled[1], (2 * logs[0][1] + logs[1][0, 1]) / 3)
    assert torch.allclose(pooled[2], logs[0][2])


def test_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard = pd.DataFrame({"run": "r", "sentence": 0, "onset": [0.0, 0.1], "target": [0, 1], "token": [7, 8]})
    monkeypatch.setattr(module.Corpus, "heard", classmethod(lambda *_: heard))
    monkeypatch.setattr(module.Run, "discover", classmethod(lambda *_: []))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: Favourite()))
    monkeypatch.setattr(module.DataConfig, "shared", classmethod(lambda *_: DataConfig()))
    built = Pooling.of(DataRoot(tmp_path), "m", DataConfig().grouping, CPU)
    assert built.evaluator.seen == frozenset({(7, 8)})
    assert built.log_prior.shape == (WORDS,)


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
    assert logits.argmax(dim=1).tolist() == [WORDS - 1] * 4
    assert reliability == pytest.approx(0.0)


def test_echoes(tmp_path: Path) -> None:
    every = runs(tmp_path)
    tool = pooling(tmp_path)
    _, reference = tool.domains({RULER: every[:1]})
    echoes = tool.echoes(reference, every)
    assert echoes.logits.shape == (1, len(SPOKEN), WORDS)
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

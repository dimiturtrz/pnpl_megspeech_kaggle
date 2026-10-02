import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.corpus import Corpus
from megspeech.data.runs import CHANNELS
from megspeech.data.spans import Spans
from megspeech.data.split import DENSE, Split
from megspeech.data.vocabulary import WORDS
from megspeech.evaluation.report import Evaluator
from megspeech.training.augment import Augment
from megspeech.training.config import OptimConfig, RunConfig, SelectionConfig
from megspeech.training.trainer import Step, Trainer

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


def trainer(config: RunConfig) -> Trainer:
    return Trainer(config, Evaluator(CPU, min_novel=1))


def test_of(tmp_path: Path) -> None:
    corpus = Corpus(split(), {"a": split(seed=1)}, frozenset({(0, 1)}))
    made = Trainer.of(corpus, RunConfig(selection=SelectionConfig(7)), cast("DataRoot", tmp_path))
    assert made.evaluator.seen == {(0, 1)}
    assert made.evaluator.min_novel == 7
    assert made.evaluator.device == CPU
    assert made.prototypes is None


def test_model() -> None:
    model = trainer(RunConfig(optim=OptimConfig(compiled=True))).model(split())
    assert next(model.parameters()).device == CPU
    assert "_orig_mod" not in next(iter(model.state_dict()))


def test_optimizer() -> None:
    made = trainer(RunConfig(optim=OptimConfig(lr=0.01)))
    optimizer = made.optimizer(made.model(split()))
    assert optimizer.defaults["lr"] == 0.01
    assert not optimizer.defaults["fused"]


def test_step() -> None:
    made = trainer(RunConfig())
    model = made.model(split())
    optimizer = torch.optim.AdamW(model.parameters())
    before = [p.clone() for p in model.parameters()]
    value = made.step(
        Step(model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0), split().capacity(2)),
        split(),
        torch.tensor([0, 1]),
    )
    assert value.ndim == 0
    assert not value.requires_grad
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_fit() -> None:
    model, history = trainer(RunConfig(augment=Augment(0.5, 0.1), optim=OptimConfig(epochs=1, batch=4, clip=1.0))).fit(
        split(), {"a": split(seed=1)}, torch.zeros(WORDS)
    )
    assert len(history) == 1
    assert isinstance(model, nn.Module)
    with pytest.raises(FloatingPointError):
        trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, lr=math.inf))).fit(
            split(), {"a": split(seed=1)}, torch.zeros(WORDS)
        )

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.corpus import Corpus
from megspeech.data.phonemes import Pronunciation
from megspeech.data.runs import CHANNELS
from megspeech.data.spans import Spans
from megspeech.data.split import DENSE, Split
from megspeech.data.vocabulary import IGNORE, WORDS
from megspeech.evaluation.report import Evaluator
from megspeech.model.sentence import SentenceNet
from megspeech.objective.heads import DIM
from megspeech.objective.prototypes import Lexicon
from megspeech.training.anchor import Anchor
from megspeech.training.augment import Augment
from megspeech.training.auxiliary import Aligned, Joint, Pronounced, Silent
from megspeech.training.config import OptimConfig, RunConfig, SelectionConfig
from megspeech.training.ema import Ema
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


def joint(train: Split) -> Joint:
    """Every token of `train` in the lexicon and the pronunciations but its first."""
    keys = train.token.unique()[1:]
    lexicon = Lexicon(keys, torch.randn(len(keys), DIM, generator=torch.Generator().manual_seed(0)))
    spelled = (torch.arange(len(keys)) % 3).unsqueeze(-1)
    return Joint((Aligned(1.0, lexicon), Pronounced(1.0, Pronunciation(keys, spelled))))


def test_of(tmp_path: Path) -> None:
    corpus = Corpus(split(), {"a": split(seed=1)}, frozenset({(0, 1)}))
    made = Trainer.of(corpus, RunConfig(selection=SelectionConfig(7)), cast("DataRoot", tmp_path))
    assert made.evaluator.seen == {(0, 1)}
    assert made.evaluator.min_novel == 7
    assert made.evaluator.device == CPU
    assert made.prototypes is None
    assert isinstance(made.auxiliary, Silent)


def test_model() -> None:
    model = trainer(RunConfig(optim=OptimConfig(compiled=True))).model(split())
    assert isinstance(model, SentenceNet)
    assert next(model.parameters()).device == CPU
    assert "_orig_mod" not in next(iter(model.state_dict()))


def test_model_warm(tmp_path: Path) -> None:
    config = RunConfig(optim=OptimConfig(compiled=False, warm="pre"))
    trained = trainer(config).model(split())
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(1.0)
    root = DataRoot(tmp_path)
    (root.models() / "pre").mkdir(parents=True)
    torch.save(trained.state_dict(), root.models() / "pre" / "model.pt")
    made = Trainer.of(Corpus(split(), {"a": split(seed=1)}, frozenset()), config, root)
    warmed = made.model(split(seed=2))
    for key, value in trained.state_dict().items():
        assert torch.equal(warmed.state_dict()[key], value)


def test_compiled() -> None:
    model = nn.Linear(2, 2)
    assert trainer(RunConfig(optim=OptimConfig(compiled=True))).compiled(model) is model


def test_optimizer() -> None:
    made = trainer(RunConfig(optim=OptimConfig(lr=0.01)))
    optimizer = made.optimizer(made.model(split()), Anchor())
    assert optimizer.defaults["lr"] == 0.01
    assert not optimizer.defaults["fused"]


def test_balanced() -> None:
    y = torch.tensor([0, 0, 0, 0, 1, IGNORE])
    assert trainer(RunConfig()).balanced(y) is None
    weight = trainer(RunConfig(optim=OptimConfig(balance=0.5))).balanced(y)
    assert weight is not None
    assert weight.shape == (WORDS,)
    assert torch.allclose(weight[:2], torch.tensor([2 / 3, 4 / 3]))
    assert weight[2:].eq(0).all()


def test_state() -> None:
    """The state trains over `total` steps with the run's anchor and class weights, as `fit` builds it."""
    optim = OptimConfig(lr=0.01, anchor=0.5, balance=0.5)
    made = trainer(RunConfig(optim=optim))
    net = made.model(split())
    state = made.state(net, split(), 10)
    assert state.model is net
    assert state.schedule.state_dict()["total_steps"] == 10
    assert state.anchor == Anchor.of(0.5, 0.01, 10)
    assert state.weight is not None
    assert torch.equal(state.weight, cast(torch.Tensor, made.balanced(split().y)))
    assert state.rows == split().capacity(optim.batch)


def test_step() -> None:
    made = trainer(RunConfig())
    model = made.model(split())
    optimizer = torch.optim.AdamW(model.parameters())
    before = [p.clone() for p in model.parameters()]
    value = made.step(
        Step(
            model,
            optimizer,
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            split().capacity(2),
            Ema.of(model, 0.0),
        ),
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
    assert isinstance(model, SentenceNet)
    train = split()
    made = Trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4)), Evaluator(CPU, min_novel=1), auxiliary=joint(train))
    net, _ = made.fit(train, {"a": split(seed=1)}, torch.zeros(WORDS))
    assert isinstance(net, SentenceNet)
    anchored = trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, anchor=0.5)))
    assert isinstance(anchored.fit(split(), {"a": split(seed=1)}, torch.zeros(WORDS))[0], SentenceNet)
    balanced = trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, balance=0.5)))
    assert isinstance(balanced.fit(split(), {"a": split(seed=1)}, torch.zeros(WORDS))[0], SentenceNet)
    with pytest.raises(FloatingPointError):
        trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, lr=math.inf))).fit(
            split(), {"a": split(seed=1)}, torch.zeros(WORDS)
        )

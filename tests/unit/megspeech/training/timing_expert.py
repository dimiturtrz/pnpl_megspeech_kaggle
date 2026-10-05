from pathlib import Path

import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import DENSE, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.model.context.timing import TimingHead
from megspeech.model.context.timing_only import TimingOnly
from megspeech.training.timing_expert import FILE, TimingExpert

CPU = torch.device("cpu")
SAMPLES = 50


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, np.arange(n) % 3, Spans.by_gap(onset), CPU)


def test_load(tmp_path: Path) -> None:
    """An empty name loads nothing; a saved head loads as it was."""
    root = DataRoot(tmp_path)
    head = TimingHead(VOCAB_SIZE)
    TimingExpert.save(head, root.models() / "t")
    assert TimingExpert.load(root, "") == {}
    loaded = TimingExpert.load(root, "t")
    assert all(torch.equal(loaded[k], v) for k, v in head.state_dict().items())


def test_save(tmp_path: Path) -> None:
    TimingExpert.save(TimingHead(VOCAB_SIZE), tmp_path / "new")
    assert (tmp_path / "new" / FILE).is_file()


def test_loss() -> None:
    """Cross-entropy over the labelled words of the drawn spans, finite and positive."""
    train = split()
    value = TimingExpert().loss(TimingOnly(VOCAB_SIZE), train, torch.arange(len(train.index)))
    assert value.item() > 0
    assert torch.isfinite(value)


def test_fit() -> None:
    """Each epoch is reported, and the model returned holds the best epoch's weights."""
    train = split()
    expert = TimingExpert(epochs=2, batch=2)
    model, history = expert.fit(train, {"a": split(seed=1)}, Evaluator(CPU, min_novel=1), train.log_prior())
    assert [row.epoch for row in history] == [0, 1]
    assert isinstance(model, TimingOnly)

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.objective.term import ContextTerm, DistilledTerm
from megspeech.training import teacher as module
from megspeech.training.config import OptimConfig
from megspeech.training.teacher import Ensembled, Teacher, Untaught

CPU = torch.device("cpu")
WORDS = 4


def train() -> Split:
    x = np.zeros((WORDS, CHANNELS, 4), dtype=np.float32)
    return Split.from_arrays(x, np.arange(WORDS) % 3, Spans.by_gap(np.arange(WORDS) * 0.2), CPU)


def test_of(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert isinstance(Teacher.of(OptimConfig(), root), Untaught)
    assert Teacher.of(OptimConfig(teacher="ens", distil=0.3, temperature=2.0), root) == Ensembled(
        root.models(), "ens", 0.3, 2.0
    )


def test_untaught_taught() -> None:
    context = ContextTerm(0.1, None)
    assert Untaught().taught(context, train(), Evaluator(CPU, min_novel=1)) is context


def test_ensembled_taught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The teacher's context head is read over every training word, in order, and paired with its label."""
    taught = torch.from_numpy(np.random.default_rng(0).normal(size=(WORDS, VOCAB_SIZE)))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: nn.Identity()))
    monkeypatch.setattr(Evaluator, "logits", lambda *_: {Head.CONTEXT: taught, Head.WORD: -taught})
    split = train()
    term = Ensembled(tmp_path, "ens", 0.3).taught(ContextTerm(0.1, None), split, Evaluator(CPU, min_novel=1))
    assert isinstance(term, DistilledTerm)
    assert (term.smoothing, term.share) == (0.1, 0.3)
    assert term.taught is taught
    assert term.y is split.y

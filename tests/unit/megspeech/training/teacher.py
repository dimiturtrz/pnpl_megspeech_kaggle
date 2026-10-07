from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig, Val
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.objective.term import ContextTerm, DistilledTerm
from megspeech.training import teacher as module
from megspeech.training.config import OptimConfig
from megspeech.training.teacher import CrossFitted, Ensembled, Teacher, Untaught

CPU = torch.device("cpu")
WORDS = 4


def train() -> Split:
    x = np.zeros((WORDS, CHANNELS, 4), dtype=np.float32)
    return Split.from_arrays(x, np.arange(WORDS) % 3, Spans.by_gap(np.arange(WORDS) * 0.2), CPU)


def test_of(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    data, words = DataConfig(), pd.DataFrame()
    assert isinstance(Teacher.of(OptimConfig(), root, data, words), Untaught)
    assert Teacher.of(OptimConfig(teacher="ens", distil=0.3, temperature=2.0), root, data, words) == Ensembled(
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


def test_over() -> None:
    whole = Ensembled(Path("m"), "ens", 0.3)
    teacher = CrossFitted.over(whole, 2, pd.Series([0, 1, 1, 0]))
    assert teacher.held.tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError, match="in no crossfit fold"):
        CrossFitted.over(whole, 2, pd.Series([0, None, 1, 0]))


def test_members() -> None:
    teacher = CrossFitted.over(Ensembled(Path("m"), "ens", 0.3), 3, pd.Series([0, 1, 2, 0]))
    assert [m.name for m in teacher.members()] == ["ens-f0", "ens-f1", "ens-f2"]


def test_cross_fitted_taught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each word's target is its held-out fold's model's row, and no other model's."""
    reads = {f"ens-f{j}": torch.full((WORDS, VOCAB_SIZE), float(j)) for j in range(3)}
    monkeypatch.setattr(Ensembled, "read", lambda self, *_: reads[self.name])
    split = train()
    teacher = CrossFitted.over(Ensembled(tmp_path, "ens", 0.3, 2.0), 3, pd.Series([2, 0, 1, 2]))
    term = teacher.taught(ContextTerm(0.1, None), split, Evaluator(CPU, min_novel=1))
    assert isinstance(term, DistilledTerm)
    assert term.taught[:, 0].tolist() == [2.0, 0.0, 1.0, 2.0]
    assert (term.smoothing, term.share, term.temperature) == (0.1, 0.3, 2.0)
    assert term.y is split.y
    with pytest.raises(ValueError, match="word folds against"):
        CrossFitted.over(Ensembled(tmp_path, "ens", 0.3), 3, pd.Series([0, 1])).taught(
            ContextTerm(0.1, None), split, Evaluator(CPU, min_novel=1)
        )


def test_of_crossfitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`teacher_folds` > 1 maps each word's run to the fold of its session; a run in no fold is refused."""
    root = DataRoot(tmp_path)
    runs = [Run(tmp_path / f"sub-0_ses-{s}_task-Sherlock1_run-1_meg.h5") for s in range(1, 5)]
    monkeypatch.setattr(module.Run, "everywhere", classmethod(lambda *_: runs))
    optim = OptimConfig(teacher="ens", teacher_folds=2)
    words = pd.DataFrame({"run": [r.name for r in runs]})
    teacher = Teacher.of(optim, root, DataConfig(val=Val.OFFICIAL, tasks=("Sherlock1",)), words)
    assert isinstance(teacher, CrossFitted)
    assert teacher.held.tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="in no crossfit fold"):
        Teacher.of(
            optim, root, DataConfig(val=Val.OFFICIAL), pd.DataFrame({"run": ["sub-0_ses-9_task-Sherlock1_run-1"]})
        )


def test_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The context head's logits over the training words, not the word head's."""
    context = torch.ones((WORDS, VOCAB_SIZE))
    monkeypatch.setattr(module.Ensemble, "load", classmethod(lambda *_: nn.Identity()))
    monkeypatch.setattr(Evaluator, "logits", lambda *_: {Head.CONTEXT: context, Head.WORD: -context})
    assert Ensembled(tmp_path, "ens", 0.3).read(train(), Evaluator(CPU, min_novel=1)) is context

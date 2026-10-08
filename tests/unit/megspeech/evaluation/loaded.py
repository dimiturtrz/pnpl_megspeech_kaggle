from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.kaggle import Track
from megspeech.data.cache.run_cache import WORD_TABLE as TABLE
from megspeech.data.config import DataConfig, Val
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation import loaded as module
from megspeech.evaluation.loaded import LoadedModel
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
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ranks = -torch.arange(VOCAB_SIZE, dtype=torch.float32)
        rows = torch.where(subject[:, None] == 0, ranks, ranks.flip(0))
        logits = nn.functional.pad(rows, (0, 0, 0, 1))[slot]
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


def loaded(tmp_path: Path, reading: Reading = Reading.OWN) -> LoadedModel:
    evaluator = Evaluator(CPU, min_novel=1)
    return LoadedModel(
        DataRoot(tmp_path),
        DataConfig(val=Val.OFFICIAL),
        CPU,
        reading,
        model=Favourite(),
        evaluator=evaluator,
        log_prior=torch.zeros(VOCAB_SIZE),
    )


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


def test_probabilities_of(tmp_path: Path) -> None:
    made = loaded(tmp_path).probabilities_of(runs(tmp_path), 0.5)
    assert RULER in made.domains
    assert made.tau == 0.5


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


def test_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Kaggle test row is one word of the split, grouped by sentence."""
    frame = pd.DataFrame(
        {"index": range(3), "subject": 0, "source": ["sentence", "sentence", "word"], "epoch": 0, "onset_s": 0.0}
    )
    monkeypatch.setattr(module.Track, "kaggle", lambda *_: type("K", (), {"metadata_frame": lambda _self: frame})())
    monkeypatch.setattr(module.Corpus, "kaggle", lambda *_: Windows.stacked(torch.zeros(3, CHANNELS, 4, device=CPU)))
    split = loaded(tmp_path).kaggle(Track.DEEP)
    assert len(split.y) == 3
    assert split.index.shape[0] == 2


def test_adapted_on(tmp_path: Path) -> None:
    tool = loaded(tmp_path)
    assert tool.adapted_on == 0
    assert replace(tool, data=replace(tool.data, subjects=(0, 1), adapt=12)).adapted_on == 12


def test_run_context(tmp_path: Path) -> None:
    split, context = loaded(tmp_path).run_context(runs(tmp_path)[POOLED])
    assert len(split.y) == len(context) == len(SPOKEN)
    assert context.argmax(dim=1).tolist() == [VOCAB_SIZE - 1] * len(SPOKEN)


def test_chosen(tmp_path: Path) -> None:
    domains, _ = loaded(tmp_path).domains({RULER: runs(tmp_path)[:1]})
    chosen = loaded(tmp_path).chosen(domains[RULER])
    assert chosen.shape == (len(SPOKEN), VOCAB_SIZE)
    assert torch.allclose(chosen.exp().sum(dim=1), torch.ones(len(SPOKEN)), atol=1e-5)


def kaggle_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {"index": range(3), "subject": 0, "source": ["sentence", "sentence", "word"], "epoch": 0, "onset_s": 0.0}
    )
    monkeypatch.setattr(module.Track, "kaggle", lambda *_: type("K", (), {"metadata_frame": lambda _self: frame})())
    monkeypatch.setattr(module.Corpus, "kaggle", lambda *_: Windows.stacked(torch.zeros(3, CHANNELS, 4, device=CPU)))


def test_kaggle_chosen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kaggle_stub(monkeypatch)
    split, chosen = loaded(tmp_path).kaggle_chosen(Track.DEEP)
    assert (len(split.y), chosen.shape) == (3, (3, VOCAB_SIZE))


def test_finished(tmp_path: Path) -> None:
    """The adjustment moves mass off the prior's favourite and the rows stay distributions."""
    prior = torch.zeros(VOCAB_SIZE)
    prior[0] = 1.0
    tool = replace(loaded(tmp_path), log_prior=prior)
    flat = torch.zeros(2, VOCAB_SIZE)
    adjusted = tool.finished(flat, 1.0)
    assert np.allclose(adjusted.sum(axis=1), 1, atol=1e-5)
    assert adjusted[0, 0] < adjusted[0, 1]
    assert np.allclose(tool.finished(flat, 0.0), 1 / VOCAB_SIZE)


def test_kaggle_probabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kaggle_stub(monkeypatch)
    tool = loaded(tmp_path)
    split, _ = tool.kaggle_chosen(Track.DEEP)
    assert np.allclose(tool.kaggle_probabilities(Track.DEEP, 0.5), tool.probabilities(split, 0.5))

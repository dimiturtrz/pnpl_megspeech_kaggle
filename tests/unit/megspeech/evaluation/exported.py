from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import SpanInput
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.kaggle import Track
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation import exported as module
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded import CPU, SPOKEN, Favourite, loaded, runs


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
    def forward(self, spans: SpanInput) -> tuple[torch.Tensor, torch.Tensor]:
        context, _ = super().forward(spans)
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

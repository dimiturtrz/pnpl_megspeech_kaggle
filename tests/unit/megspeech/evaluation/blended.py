from dataclasses import replace
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import nn

from megspeech.data.batching.split import Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.evaluation.blended import Blended
from megspeech.evaluation.loaded import LoadedModel
from tests.unit.megspeech.evaluation.loaded import POOLED, SPOKEN, kaggle_stub, loaded, runs


class Flat(nn.Module):
    """Every word equally likely."""

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(slot.shape[0], slot.shape[1], VOCAB_SIZE)
        return logits, logits


class Shuffled(LoadedModel):
    """Reads other words than its peers."""

    @override
    def run_context(self, run: Run) -> tuple[Split, torch.Tensor]:
        split, context = super().run_context(run)
        return replace(split, y=split.y + 1), context


def pair(tmp_path: Path, *, second: type[LoadedModel] = LoadedModel) -> Blended:
    first = replace(loaded(tmp_path), names=("a", "b"), log_prior=torch.zeros(VOCAB_SIZE))
    other = loaded(tmp_path)
    flat = second(
        other.root,
        other.data,
        other.device,
        other.reading,
        model=Flat(),
        evaluator=replace(other.evaluator, seen=frozenset({(1, 2)})),
        log_prior=torch.ones(VOCAB_SIZE),
        names=("c",),
    )
    return Blended.of([first, flat])


def test_of(tmp_path: Path) -> None:
    """Weights count members, the prior is their weighted mean, and a word is seen if any group saw it."""
    blended = pair(tmp_path)
    assert blended.weights == pytest.approx((2 / 3, 1 / 3))
    assert blended.names == ("a", "b", "c")
    assert torch.allclose(blended.log_prior, torch.full((VOCAB_SIZE,), 1 / 3))
    assert blended.evaluator.seen == frozenset({(1, 2)})
    assert blended.model is blended.groups[0].model


def test_adapted_on(tmp_path: Path) -> None:
    blended = pair(tmp_path)
    adapted = replace(blended.groups[1], data=replace(blended.data, subjects=(0, 1), adapt=12))
    assert replace(blended, groups=(blended.groups[0], adapted)).adapted_on == 12
    assert blended.adapted_on == 0


def test_blend(tmp_path: Path) -> None:
    """Each group's log-probabilities enter at its member-count weight."""
    blended = pair(tmp_path)
    run = runs(tmp_path)[POOLED]
    _, context = blended.blend(lambda group: group.run_context(run), run.name)
    ranked = torch.log_softmax(-torch.arange(VOCAB_SIZE, dtype=torch.float32).flip(0), dim=-1)
    flat = torch.full((VOCAB_SIZE,), -float(np.log(VOCAB_SIZE)))
    assert context.shape == (len(SPOKEN), VOCAB_SIZE)
    assert torch.allclose(context[0], 2 / 3 * ranked + 1 / 3 * flat, atol=1e-5)


def test_blend_refuses_different_words(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different words for"):
        run = runs(tmp_path)[POOLED]
        pair(tmp_path, second=Shuffled).blend(lambda group: group.run_context(run), run.name)


def test_run_context(tmp_path: Path) -> None:
    split, context = pair(tmp_path).run_context(runs(tmp_path)[POOLED])
    assert len(split.y) == len(context) == len(SPOKEN)


def test_kaggle_chosen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kaggle_stub(monkeypatch)
    blended = pair(tmp_path)
    split, chosen = blended.kaggle_chosen(Track.BROAD)
    assert (len(split.y), chosen.shape) == (3, (3, VOCAB_SIZE))
    assert blended.kaggle_probabilities(Track.BROAD, 0.0).shape == (3, VOCAB_SIZE)


@pytest.mark.parametrize("method", ["context", "splits", "domains", "kaggle"])
def test_single_group_reads_refused(tmp_path: Path, method: str) -> None:
    blended = pair(tmp_path)
    argument = {"context": None, "splits": [], "domains": {}, "kaggle": Track.BROAD}[method]
    with pytest.raises(NotImplementedError, match="run_context and kaggle_chosen only"):
        getattr(blended, method)(argument)

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
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.exported import kaggle_stub
from tests.unit.megspeech.evaluation.loaded import POOLED, SPOKEN, loaded, runs


class Flat(nn.Module):
    """Every word equally likely."""

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(slot.shape[0], slot.shape[1], VOCAB_SIZE)
        return logits, logits


class Shuffled(ExportedModel):
    """Reads other words than its peers."""

    @override
    def run_context(self, run: Run) -> tuple[Split, torch.Tensor]:
        split, context = super().run_context(run)
        return replace(split, y=split.y + 1), context

    @override
    def domain_logits(self, runs: list[Run]) -> tuple[dict[str, Split], dict[str, dict[Head, torch.Tensor]]]:
        domains, logits = super().domain_logits(runs)
        return {name: replace(split, y=split.y + 1) for name, split in domains.items()}, logits


def pair(tmp_path: Path, *, second: type[ExportedModel] = ExportedModel) -> Blended:
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


def test_agree(tmp_path: Path) -> None:
    split, _ = loaded(tmp_path).run_context(runs(tmp_path)[POOLED])
    assert Blended.agree(split, split)
    assert not Blended.agree(split, replace(split, y=split.y + 1))


def test_domain_logits(tmp_path: Path) -> None:
    """A blended report weights each group's log-probabilities at its member count, in every domain and head."""
    blended = pair(tmp_path)
    domains, logits = blended.domain_logits(runs(tmp_path))
    alone, own = blended.groups[0].domain_logits(runs(tmp_path))
    assert RULER in domains
    flat = -float(np.log(VOCAB_SIZE))
    for head in Head:
        assert torch.allclose(logits[RULER][head], 2 / 3 * own[RULER][head] + 1 / 3 * flat, atol=1e-5)
    assert alone.keys() == domains.keys()
    assert blended.report(runs(tmp_path)).macro.n == blended.groups[0].report(runs(tmp_path)).macro.n


def test_domain_logits_refuses_different_words(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different words for the scored domains"):
        pair(tmp_path, second=Shuffled).domain_logits(runs(tmp_path))


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
    argument: object = {"context": None, "splits": [], "domains": {}, "kaggle": Track.BROAD}[method]
    with pytest.raises(NotImplementedError, match="run_context and kaggle_chosen only"):
        getattr(blended, method)(argument)

import math
from dataclasses import replace
from pathlib import Path
from typing import override

import pandas as pd
import pytest
import torch
from torch import nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.uncertainty import POOLED, Decomposition, Members, Tally
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded import SPOKEN, Favourite, loaded, runs


class Reversed(Favourite):
    """`Favourite` ranking the context vocabulary the other way: it disagrees with `Favourite` on every word."""

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, word = super().forward(x, onset, slot, subject)
        return context.flip(-1), word


class Pair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.members = nn.ModuleList([Favourite(), Reversed()])


def members(tmp_path: Path) -> Members:
    return Members(replace(loaded(tmp_path), model=Pair()))


def test_entropy() -> None:
    """A uniform distribution over `v` words has entropy ln v; a one-hot has none."""
    uniform = torch.full((1, VOCAB_SIZE), 1 / VOCAB_SIZE)
    assert Decomposition.entropy(uniform).item() == pytest.approx(math.log(VOCAB_SIZE), rel=1e-5)
    assert Decomposition.entropy(torch.eye(VOCAB_SIZE)[:1]).item() == 0


def test_of() -> None:
    """Identical members have no epistemic part; confident members disagreeing have only it, ln 2 for a pair."""
    first = torch.full((1, 1, VOCAB_SIZE), -1e4)
    second = first.clone()
    first[..., 0], second[..., 1] = 0, 0
    same = Decomposition.of(torch.cat([first, first]))
    assert same.epistemic.item() == pytest.approx(0, abs=1e-6)
    apart = Decomposition.of(torch.cat([first, second]))
    assert apart.aleatoric.item() == pytest.approx(0, abs=1e-6)
    assert apart.epistemic.item() == pytest.approx(math.log(2), rel=1e-5)
    assert apart.total.item() == pytest.approx(apart.epistemic.item() + apart.aleatoric.item())


def test_networks(tmp_path: Path) -> None:
    assert len(members(tmp_path).networks) == 2


def test_logits(tmp_path: Path) -> None:
    domains, _ = members(tmp_path).loaded.domains({RULER: runs(tmp_path)[:1]})
    logits = members(tmp_path).logits(domains[RULER])
    assert logits.shape == (2, len(SPOKEN), VOCAB_SIZE)
    assert logits.exp().sum(-1).allclose(torch.ones(2, len(SPOKEN)), atol=1e-4)


def test_hits(tmp_path: Path) -> None:
    """The favourite ranks words 0-9 on top, the reversed member the last ten; the ensemble's row comes last."""
    domains, _ = members(tmp_path).loaded.domains({RULER: runs(tmp_path)[:1]})
    logits = members(tmp_path).logits(domains[RULER])
    hits = members(tmp_path).hits(logits, torch.tensor([0, 1, 2, 3, 4, 5]))
    assert hits.shape == (3, len(SPOKEN))
    assert hits[0].all()
    assert not hits[1].any()


def test_frame(tmp_path: Path) -> None:
    """Every scored word is a row; the members' disagreement is all epistemic."""
    domains, _ = members(tmp_path).loaded.domains({RULER: runs(tmp_path)[:1]})
    frame, hits = members(tmp_path).frame(domains[RULER], RULER)
    assert len(frame) == len(SPOKEN) == hits.shape[1]
    assert (frame["epistemic"] > 0).all()
    assert (frame["domain"] == RULER).all()


def test_partition(tmp_path: Path) -> None:
    frame, hits = members(tmp_path).partition(runs(tmp_path))
    assert RULER in hits
    assert len(frame) == sum(h.shape[1] for h in hits.values())


def tally() -> Tally:
    frame = pd.DataFrame(
        {
            "domain": "d",
            "label": [0, 1, 2, 3],
            "hit": [True, True, False, False],
            "total": [1.0, 1.0, 3.0, 3.0],
            "aleatoric": [0.5, 0.5, 2.0, 2.0],
            "epistemic": [0.5, 0.5, 1.0, 1.0],
        }
    )
    hits = torch.tensor([[True, True, False, False], [True, False, False, False], [True, True, False, False]])
    return Tally(frame, {"d": hits})


def test_entropies() -> None:
    assert tally().entropies(tally().frame) == "n=4 total=2.000 aleatoric=1.250 epistemic=0.750 epistemic/total=0.375"


def test_accuracy() -> None:
    line = tally().accuracy("d")
    assert "ensemble=0.5000" in line
    assert "members=2 n=4" in line


def test_lines() -> None:
    lines = tally().lines()
    assert any(line.startswith(f"{POOLED} correct: n=2") for line in lines)
    assert any(line.startswith("d wrong: n=2") for line in lines)
    assert any(line.startswith("d members bacc10") for line in lines)

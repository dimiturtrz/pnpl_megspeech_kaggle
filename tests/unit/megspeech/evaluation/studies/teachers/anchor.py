import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.evaluation.studies.teachers.alignment import Lexical, Words
from megspeech.evaluation.studies.teachers.anchor import Anchor, Reachable
from megspeech.objective.targets.t5 import Lexicon

ROWS = 400


def pair(shared: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Two views whose first `shared` coordinates carry one signal and the rest are noise."""
    generator = torch.Generator().manual_seed(0)
    signal = torch.randn(ROWS, shared, generator=generator)
    x = torch.cat([signal, torch.randn(ROWS, 4, generator=generator)], dim=1)
    y = torch.cat([signal, torch.randn(ROWS, 6, generator=generator)], dim=1)
    return x, y


def anchor() -> Anchor:
    x, y = pair()
    table = pd.DataFrame({"run": "r", "position": np.arange(ROWS), "token": np.arange(ROWS)})
    every, later = Lexicon(torch.arange(1, ROWS), y[1:]), Lexicon(torch.arange(2, ROWS), y[2:])
    return Anchor.fit(Words(table, x), [Lexical(every), Lexical(later)], k=3)


def test_reachable_fit() -> None:
    """The shared directions keep their weight; the noise ones fade."""
    x, y = pair()
    reach = Reachable.fit(x, y, 3)
    spread = reach.projected(y).std(0)
    assert reach.axes.shape == (8, 3)
    assert spread[:2].min() > 0.9
    assert spread[2] < 0.5 * spread[:2].min()


def test_projected() -> None:
    x, y = pair()
    reach = Reachable.fit(x, y, 2)
    assert torch.allclose(reach.projected(y).mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-9)


def test_anchor_fit() -> None:
    fused = anchor()
    assert fused.vectors.shape == (ROWS, 6)
    assert not fused.present[0]
    assert fused.present[1:].all()
    assert np.all(fused.vectors[1, 3:] == 0)
    assert np.any(fused.vectors[1, :3] != 0)


def test_present() -> None:
    assert Anchor(pd.DataFrame(), np.array([[1.0, np.nan], [1.0, 2.0]])).present.tolist() == [False, True]


def test_runs() -> None:
    table = pd.DataFrame({"run": ["a", "a", "b"], "position": [1, 0, 0]})
    placed = Anchor(table, np.array([[1.0], [2.0], [3.0]])).runs({"a": 2, "b": 1})
    assert placed["a"].vectors.tolist() == [[2.0], [1.0]]
    assert placed["b"].vectors.tolist() == [[3.0]]


def test_runs_refuses_partial_run() -> None:
    with pytest.raises(ValueError, match="positions would shift"):
        Anchor(pd.DataFrame({"run": ["a"], "position": [0]}), np.array([[1.0]])).runs({"a": 2})

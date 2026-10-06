import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.objective.auxiliary.adversarial import Adversary, TimingTerm

CPU = torch.device("cpu")
WORDS = pd.DataFrame({"run": ["a"] * 4, "sentence": [0, 0, 0, 1], "onset": [0.0, 0.2, 0.8, 0.1]})


def term() -> TimingTerm:
    return TimingTerm.of(1.0, WORDS, CPU)


def test_adversary() -> None:
    assert Adversary(6)(torch.zeros(2, 3, 6)).shape == (2, 3, 2)


def test_of() -> None:
    """Position counts within the sentence; a sentence's last word has no duration; both standardised."""
    timing = term()
    assert timing.known[:, 0].tolist() == [True, True, False, False]
    assert timing.known[:, 1].all()
    assert timing.targets[:, 1].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert timing.targets[:, 0][timing.known[:, 0]].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert timing.targets[0, 1] < timing.targets[2, 1]
    assert timing.targets[2, 0] == 0.0


def test_head() -> None:
    assert isinstance(term().head(6), Adversary)


def split(n: int) -> Split:
    x = np.zeros((n, 4, 4), dtype=np.float32)
    return Split.from_arrays(x, np.arange(n) % 3, Spans.by_gap(np.arange(n) * 2.0), CPU)


def test_labels() -> None:
    assert torch.equal(term().labels(split(4)), torch.arange(4))
    with pytest.raises(ValueError, match="timing rows"):
        term().labels(split(3))


def test_unweighted() -> None:
    """Perfect predictions cost nothing, the mean costs the known entries' mean square, empty slots cost nothing."""
    timing = term()
    target = torch.tensor([[0, 1, 2, IGNORE]])
    exact = timing.targets[[0, 1, 2]][None]
    exact = torch.cat([exact, torch.zeros(1, 1, 2)], dim=1)
    assert timing.unweighted(Adversary(2), exact, target).item() == pytest.approx(0.0)
    zero = timing.unweighted(Adversary(2), torch.zeros(1, 4, 2), target)
    kept = timing.targets[[0, 1, 2]][timing.known[[0, 1, 2]]]
    assert zero.item() == pytest.approx(kept.square().mean().item())


def test_weighted() -> None:
    """Zero predictions cost the known entries' weighted mean square; weight 1 is the unweighted loss."""
    timing = term()
    target = torch.tensor([[0, 1, 2, IGNORE]])
    zero = torch.zeros(1, 4, 2)
    assert torch.equal(
        timing.weighted(Adversary(2), zero, target, torch.ones(1, 4)), timing.unweighted(Adversary(2), zero, target)
    )
    rows = torch.tensor([[1.0, 3.0, 1.0, 1.0]])
    square = timing.targets[[0, 1, 2]].square() * timing.known[[0, 1, 2]]
    weight = rows[0, :3, None] * timing.known[[0, 1, 2]]
    assert timing.weighted(Adversary(2), zero, target, rows).item() == pytest.approx(
        ((square * rows[0, :3, None]).sum() / weight.sum()).item()
    )

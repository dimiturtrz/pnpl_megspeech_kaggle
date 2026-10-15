import math

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.objective.auxiliary.adversary.adversarial import Adversary, AdversaryTarget, RunTerm, TimingTerm
from megspeech.objective.term import Reading

CPU = torch.device("cpu")
WORDS = pd.DataFrame({"run": ["a"] * 4, "sentence": [0, 0, 0, 1], "onset": [0.0, 0.2, 0.8, 0.1]})


def term() -> TimingTerm:
    return TimingTerm.of(1.0, WORDS, CPU)


def test_adversary() -> None:
    assert Adversary(6)(torch.zeros(2, 3, 6)).shape == (2, 3, 2)


def test_timing_term_of() -> None:
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
    with pytest.raises(ValueError, match="adversary rows"):
        term().labels(split(3))


def test_timing_term_unweighted() -> None:
    """Perfect predictions cost nothing, the mean costs the known entries' mean square, empty slots cost nothing."""
    timing = term()
    target = torch.tensor([[0, 1, 2, IGNORE]])
    exact = timing.targets[[0, 1, 2]][None]
    exact = torch.cat([exact, torch.zeros(1, 1, 2)], dim=1)
    assert timing.unweighted(timing.read(Adversary(2), exact, target)).item() == pytest.approx(0.0)
    zero = timing.unweighted(timing.read(Adversary(2), torch.zeros(1, 4, 2), target))
    kept = timing.targets[[0, 1, 2]][timing.known[[0, 1, 2]]]
    assert zero.item() == pytest.approx(kept.square().mean().item())


def test_unweighted_rows() -> None:
    """Zero predictions cost the known entries' weighted mean square; weight 1 is the unweighted loss."""
    timing = term()
    target = torch.tensor([[0, 1, 2, IGNORE]])
    zero = torch.zeros(1, 4, 2)
    assert torch.equal(
        timing.unweighted(Reading(Adversary(2), zero, target, torch.ones(1, 4))),
        timing.unweighted(timing.read(Adversary(2), zero, target)),
    )
    rows = torch.tensor([[1.0, 3.0, 1.0, 1.0]])
    square = timing.targets[[0, 1, 2]].square() * timing.known[[0, 1, 2]]
    weight = rows[0, :3, None] * timing.known[[0, 1, 2]]
    assert timing.unweighted(Reading(Adversary(2), zero, target, rows)).item() == pytest.approx(
        ((square * rows[0, :3, None]).sum() / weight.sum()).item()
    )


RUNS = pd.DataFrame({"run": ["b", "a", "b", "c"]})


def against_runs() -> RunTerm:
    return RunTerm.of(0.5, RUNS, CPU)


def test_adversary_outputs() -> None:
    assert Adversary(6, 5)(torch.zeros(2, 3, 6)).shape == (2, 3, 5)


def test_run_term_of() -> None:
    """Each word's run indexed among the sorted distinct runs; nothing tallied yet."""
    runs = against_runs()
    assert (runs.weight, runs.classes, runs.runs.tolist()) == (0.5, 3, [1, 0, 1, 2])
    assert runs.hits.tolist() == [0.0, 0.0]


def test_timing_term_outputs() -> None:
    assert term().outputs() == 2


def test_run_term_outputs() -> None:
    assert against_runs().outputs() == 3


def test_timing_term_rows() -> None:
    assert term().rows() == 4


def test_run_term_rows() -> None:
    assert against_runs().rows() == 4


def test_run_term_head() -> None:
    assert against_runs().head(6)(torch.zeros(1, 2, 6)).shape == (1, 2, 3)


def test_run_term_labels() -> None:
    assert torch.equal(against_runs().labels(split(4)), torch.arange(4))


def test_run_term_unweighted() -> None:
    """Confident right logits cost ~0, uniform ones ln 3; empty slots are neither scored nor tallied."""
    runs = against_runs()
    target = torch.tensor([[0, 1, 3, IGNORE]])
    right = torch.full((1, 4, 3), -20.0)
    right[0, 0, 1] = right[0, 1, 0] = right[0, 2, 2] = 20.0
    assert runs.unweighted(runs.read(Adversary(2, 3), right, target)).item() == pytest.approx(0.0, abs=1e-6)
    assert runs.hits.tolist() == [3.0, 3.0]
    uniform = runs.unweighted(runs.read(Adversary(2, 3), torch.zeros(1, 4, 3), target))
    assert uniform.item() == pytest.approx(math.log(3))


def test_drained() -> None:
    """The tallied accuracy against the majority run's share, then emptied."""
    runs = against_runs()
    wrong = torch.zeros(1, 2, 3)
    wrong[0, :, 2] = 1.0
    runs.unweighted(runs.read(Adversary(2, 3), wrong, torch.tensor([[0, 3]])))
    assert runs.drained() == {"accuracy": 0.5, "majority": 0.5, "classes": 3.0}
    assert runs.hits.tolist() == [0.0, 0.0]
    assert runs.drained()["accuracy"] == 0.0


def test_term() -> None:
    assert isinstance(AdversaryTarget.TIMING.term(WORDS, CPU), TimingTerm)
    run = AdversaryTarget.RUN.term(WORDS, CPU)
    assert isinstance(run, RunTerm)
    assert (run.weight, run.classes) == (1.0, 1)

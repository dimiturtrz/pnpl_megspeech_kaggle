from types import SimpleNamespace
from typing import cast, override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head, Score
from megspeech.evaluation.studies.gap import Condition, Grid, Locator, Subset

CPU = torch.device("cpu")
SPAN = 4
SPANS = 6
CLASSES = 5
LOGIT = 10


def split(n: int = SPAN * SPANS) -> Split:
    """Spans of four words close in time, far from each other; each window holds its word's label."""
    y = np.arange(n) % CLASSES
    onset = np.arange(n) % SPAN * 0.2 + np.arange(n) // SPAN * 20.0
    x = torch.from_numpy(y.astype(np.float32))[:, None, None].expand(n, 1, 2).contiguous()
    return Split.of(Windows.stacked(x, centred=False), (y, y), Spans.by_gap(onset), np.ones(n))


class Reader(nn.Module):
    """Reads its own word's label off the window only, ignoring context and onsets."""

    @override
    def forward(
        self, x: torch.Tensor, onset: torch.Tensor, slot: torch.Tensor, subject: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.where(slot < len(x), slot, 0)
        logits = torch.nn.functional.one_hot(x[rows, 0, 0].long(), VOCAB_SIZE).float() * LOGIT
        return logits, logits


class Stub:
    def __init__(self) -> None:
        self.model = Reader()
        self.evaluator = Evaluator(CPU, min_novel=1)
        self.log_prior = torch.zeros(VOCAB_SIZE)

    def context(self, split: Split) -> Tensor:
        return self.evaluator.logits(self.model, split)[Head.CONTEXT]

    def score(self, context: Tensor, y: Tensor) -> Score:
        return self.evaluator.evaluate(context, y, self.log_prior)

    def ruled(self, runs: list[Run]) -> dict[str, list[Run]]:
        return {"d": runs}

    def domains(self, held: dict[str, list[Run]]) -> tuple[dict[str, Split], None]:
        return {name: split() for name in held}, None

    def report(self, runs: list[Run]) -> SimpleNamespace:
        return SimpleNamespace(tau=0.5)


def locator() -> Locator:
    return Locator(cast(LoadedModel, Stub()), split(), {"d": split()}, 0.0, 0)


def grid() -> Grid:
    return Grid(
        {
            Subset.FIT: {Condition.REAL: 0.9, Condition.MEG: 0.3, Condition.ORDER: 0.5},
            Subset.VAL: {Condition.REAL: 0.6, Condition.MEG: 0.3, Condition.ORDER: 0.5},
        }
    )


def test_of() -> None:
    found = Locator.of(cast(LoadedModel, Stub()), split(), [cast(Run, "run")], 0)
    assert found.tau == 0.5
    assert len(found.fitted.heard()) >= len(split().labelled())
    assert set(found.domains) == {"d"}


def test_apply() -> None:
    original = split()
    assert Condition.REAL.apply(original, 0) is original
    assert torch.equal(Condition.MEG.apply(original, 0).windows.at, original.permuted(0).windows.at)
    assert torch.equal(Condition.ORDER.apply(original, 0).index, original.ordered(0).index)


def test_drop() -> None:
    assert grid().drop(Subset.FIT, Condition.ORDER) == pytest.approx(0.4)
    assert grid().drop(Subset.VAL, Condition.REAL) == 0


def test_gap() -> None:
    assert grid().gap(Condition.REAL) == pytest.approx(0.3)
    assert grid().gap(Condition.ORDER) == 0


def test_text_excess() -> None:
    assert grid().text_excess() == pytest.approx(0.3)


def test_meg_excess() -> None:
    assert grid().meg_excess() == pytest.approx(0.3)


def test_lines() -> None:
    lines = grid().lines()
    assert lines[0].split() == [condition.value for condition in Condition]
    assert lines[1].split()[:2] == ["fit", "0.9000"]
    assert lines[-1].startswith("fit-minus-val drop")


def test_fit() -> None:
    """A model reading only its own window is blind to word order and loses to a MEG shuffle."""
    found = locator()
    assert found.fit(Condition.ORDER) == pytest.approx(found.fit(Condition.REAL))
    assert found.fit(Condition.MEG) < found.fit(Condition.REAL)


def test_val() -> None:
    found = locator()
    assert found.val(Condition.ORDER) == pytest.approx(found.val(Condition.REAL))
    assert found.val(Condition.MEG) < found.val(Condition.REAL)


def test_grid() -> None:
    scores = locator().grid().scores
    assert set(scores) == set(Subset)
    assert all(set(by_condition) == set(Condition) for by_condition in scores.values())

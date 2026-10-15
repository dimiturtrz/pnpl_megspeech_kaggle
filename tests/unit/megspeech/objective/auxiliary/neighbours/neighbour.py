import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from megspeech.data.batching.split import SpanInput, Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.model.auxiliary import Tap, WithAuxiliary
from megspeech.objective.auxiliary.neighbours.neighbour import OUTSIDE, NextWordTerm, PreviousWordTerm
from megspeech.objective.term import Reading
from tests.unit.megspeech.model.auxiliary import ONSET, SLOT, net, x

TARGET = torch.tensor([[3, OUTSIDE, 7, IGNORE], [5, 6, IGNORE, IGNORE]])


def test_head() -> None:
    assert NextWordTerm(weight=1.0).head(16)(torch.zeros(2, 3, 16)).shape == (2, 3, VOCAB_SIZE)


def test_labels() -> None:
    split = cast("Split", SimpleNamespace(y=torch.tensor([4, IGNORE, 0])))
    assert NextWordTerm(weight=1.0).labels(split).tolist() == [4, OUTSIDE, 0]


def test_beside() -> None:
    target = torch.tensor([[1, 2, 3]])
    assert NextWordTerm.beside(target, 2).tolist() == [[3, IGNORE, IGNORE]]
    assert NextWordTerm.beside(target, 0).tolist() == [[1, 2, 3]]
    assert NextWordTerm.beside(target, -1).tolist() == [[IGNORE, 1, 2]]


def test_next_word_term_shift() -> None:
    assert NextWordTerm(weight=1.0).shift(TARGET).tolist() == [
        [OUTSIDE, 7, IGNORE, IGNORE],
        [6, IGNORE, IGNORE, IGNORE],
    ]


def test_previous_word_term_shift() -> None:
    assert PreviousWordTerm(weight=1.0).shift(TARGET).tolist() == [[IGNORE, 3, OUTSIDE, 7], [IGNORE, 5, 6, IGNORE]]


def test_neighbours() -> None:
    """Outside-the-50 sources still predict; outside-the-50, absent and cross-span neighbours are ignored."""
    assert NextWordTerm(weight=1.0).neighbours(TARGET).tolist() == [
        [IGNORE, 7, IGNORE, IGNORE],
        [6, IGNORE, IGNORE, IGNORE],
    ]
    assert PreviousWordTerm(weight=1.0).neighbours(TARGET).tolist() == [
        [IGNORE, 3, IGNORE, IGNORE],
        [IGNORE, 5, IGNORE, IGNORE],
    ]


def test_unweighted() -> None:
    term = NextWordTerm(weight=2.0)
    head = term.head(4)
    uniform = torch.zeros(2, 4, VOCAB_SIZE)
    assert term.loss(term.read(head, uniform, TARGET)).item() == pytest.approx(2.0 * math.log(VOCAB_SIZE))
    assert term.unweighted(term.read(head, uniform, torch.full((2, 4), IGNORE))).item() == 0.0


def test_unweighted_rows() -> None:
    """Rows scale each predicted slot's loss; a slot without a heard neighbour contributes nothing."""
    term = NextWordTerm(weight=1.0)
    head = term.head(4)
    uniform = torch.zeros(2, 4, VOCAB_SIZE)
    flat = torch.ones(2, 4)
    assert term.unweighted(Reading(head, uniform, TARGET, flat)).item() == pytest.approx(math.log(VOCAB_SIZE))
    skewed = torch.tensor([[1.0, 9.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    assert term.unweighted(Reading(head, uniform, TARGET, skewed)).item() == pytest.approx(math.log(VOCAB_SIZE))
    assert term.unweighted(Reading(head, uniform, torch.full((2, 4), IGNORE), flat)).item() == 0.0


def test_local() -> None:
    assert NextWordTerm.local
    assert PreviousWordTerm.local


def test_gradient_skips_the_mixer() -> None:
    """Under the context tap the head still reads the local embedding: its loss reaches the encoder and the head,
    never the span transformer."""
    model = net()
    term = NextWordTerm(weight=1.0)
    head = term.readout(model.words.head, model.words.width)
    heads = torch.nn.ModuleList([head])
    auxiliary = WithAuxiliary(model, heads, Tap.CONTEXT, (Tap.WORD,))
    output = auxiliary(SpanInput(x(), ONSET, SLOT, torch.zeros(3, dtype=torch.long)))[2]
    target = torch.tensor([[1, 2, IGNORE], [4, IGNORE, IGNORE]])
    term.loss(term.read(head, output, target)).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.context.parameters())
    assert any(p.grad is not None and p.grad.any() for p in model.words.parameters())
    grad = cast("nn.Linear", head.get_submodule("1")).weight.grad
    assert grad is not None
    assert grad.any()

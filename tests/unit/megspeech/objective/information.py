from types import SimpleNamespace
from typing import cast

import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.objective.information import InformedTerm
from megspeech.objective.term import ContextTerm

LABELS = torch.tensor([0, 0, 1, 2, 2, 2])
PRIOR = torch.arange(1, VOCAB_SIZE + 1, dtype=torch.float)
PRESENT = PRIOR[:3] / PRIOR[:3].sum()
SHARP = 40.0


def onehot(predicted: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(predicted, VOCAB_SIZE).float() * SHARP


def test_of() -> None:
    base = ContextTerm(0.0, None)
    assert InformedTerm.of(base, 0.0, PRIOR) is base
    assert isinstance(InformedTerm.of(base, 0.1, PRIOR), InformedTerm)


def test_information() -> None:
    perfect = InformedTerm.information(onehot(LABELS), LABELS, PRIOR)
    assert torch.isclose(perfect, -(PRESENT * PRESENT.log2()).sum(), atol=1e-4)
    assert abs(float(InformedTerm.information(torch.zeros(len(LABELS), VOCAB_SIZE), LABELS, PRIOR))) < 1e-5


def test_information_ignores_unlabelled() -> None:
    labels = torch.cat([LABELS, torch.tensor([IGNORE, IGNORE])])
    logits = torch.cat([onehot(LABELS), onehot(torch.tensor([9, 9]))])
    assert torch.isclose(
        InformedTerm.information(logits, labels, PRIOR), InformedTerm.information(onehot(LABELS), LABELS, PRIOR)
    )


def test_information_matches_ruler_at_one_hot() -> None:
    predicted = torch.tensor([0, 1, 1, 2, 0, 2])
    soft = InformedTerm.information(onehot(predicted), LABELS, PRIOR)
    assert abs(float(soft) - Ruler.information(predicted, LABELS, PRIOR)) < 1e-4


def test_information_gradient() -> None:
    logits = torch.randn(len(LABELS), VOCAB_SIZE, generator=torch.Generator().manual_seed(0), requires_grad=True)
    InformedTerm.information(logits, LABELS, PRIOR).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_loss() -> None:
    base = ContextTerm(0.0, None)
    term = InformedTerm(base=base, prior=PRIOR, weight=0.5)
    output = onehot(LABELS)[None]
    target = LABELS[None]
    expected = base.loss(torch.nn.Identity(), output, target) + 0.5 * term.unweighted(
        torch.nn.Identity(), output, target
    )
    assert torch.isclose(term.loss(torch.nn.Identity(), output, target), expected)


def test_unweighted() -> None:
    term = InformedTerm(base=ContextTerm(0.0, None), prior=PRIOR, weight=0.5)
    value = term.unweighted(torch.nn.Identity(), onehot(LABELS)[None], LABELS[None])
    assert value < 0


def test_labels() -> None:
    split = cast("Split", SimpleNamespace(y=LABELS, unmuted=lambda y: y))
    term = InformedTerm(base=ContextTerm(0.0, None), prior=PRIOR, weight=0.5)
    assert term.labels(split).tolist() == LABELS.tolist()

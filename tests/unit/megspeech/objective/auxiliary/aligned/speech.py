from typing import cast

import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.objective.auxiliary.aligned.semantic import Semantic
from megspeech.objective.auxiliary.aligned.speech import SpeechTerm
from megspeech.objective.targets.speech import Speech
from megspeech.objective.term import Reading

WIDTH = 16
DIM = 8
GENERATOR = torch.Generator().manual_seed(0)


class Words:
    def __init__(self, n: int) -> None:
        self.y = torch.zeros(n, dtype=torch.long)


def term() -> SpeechTerm:
    return SpeechTerm(2.0, Speech(torch.tensor([0, -1, 1, 2]), torch.randn(3, DIM, generator=GENERATOR).half()))


def test_head() -> None:
    head = term().head(WIDTH)
    assert isinstance(head, Semantic)
    assert head(torch.randn(2, 3, WIDTH, generator=GENERATOR)).shape == (2, 3, DIM)


def test_labels() -> None:
    assert term().labels(cast("Split", Words(4))).tolist() == [0, -1, 1, 2]
    with pytest.raises(ValueError, match="3 words against 4"):
        term().labels(cast("Split", Words(3)))


def test_unweighted() -> None:
    """Positive over the words with a target, differentiable into the projection, 0 when none has one."""
    aligned = term()
    head = aligned.head(WIDTH)
    projected = head(torch.randn(2, 3, WIDTH, generator=GENERATOR))
    value = aligned.unweighted(aligned.read(head, projected, torch.tensor([[0, 1, -100], [2, -1, -100]])))
    assert value.ndim == 0
    assert value.item() > 0
    value.backward()
    assert cast("Semantic", head).project.weight.grad is not None
    assert aligned.unweighted(aligned.read(head, projected, torch.full((2, 3), -1))).item() == 0.0


def test_softmax() -> None:
    """The softmax contrast is selected by the flag and differentiates into the projection."""
    softmaxed = SpeechTerm(2.0, term().speech, softmax=True)
    head = softmaxed.head(WIDTH)
    projected = head(torch.randn(2, 3, WIDTH, generator=GENERATOR))
    value = softmaxed.unweighted(softmaxed.read(head, projected, torch.tensor([[0, 1, -100], [2, -1, -100]])))
    assert value.item() > 0
    value.backward()
    assert cast("Semantic", head).project.weight.grad is not None


def test_unweighted_rows() -> None:
    """Rows at weight 1 are the unweighted loss, in both contrasts; a heavier row moves it."""
    target = torch.tensor([[0, 1, -100], [2, -1, -100]])
    for softmax in (False, True):
        aligned = SpeechTerm(2.0, term().speech, softmax=softmax)
        head = aligned.head(WIDTH)
        projected = head(torch.randn(2, 3, WIDTH, generator=GENERATOR))
        assert torch.equal(
            aligned.unweighted(Reading(head, projected, target, torch.ones(2, 3))),
            aligned.unweighted(aligned.read(head, projected, target)),
        )
        skewed = aligned.unweighted(Reading(head, projected, target, torch.tensor([[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]])))
        assert skewed.item() != aligned.unweighted(aligned.read(head, projected, target)).item()

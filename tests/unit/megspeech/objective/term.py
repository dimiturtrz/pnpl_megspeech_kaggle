import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional

from megspeech.data.batching.split import DENSE, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.context import Objective
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import Classified, ContextTerm, LabelTerm, WordTerm

CPU = torch.device("cpu")
Y = torch.tensor([[3, 7], [IGNORE, 3]])


def split() -> Split:
    n = 8
    x = np.zeros((n, CHANNELS, 4), dtype=np.float32)
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, np.arange(n) % 3, Spans.by_gap(onset), CPU)


def logits() -> torch.Tensor:
    right = torch.full((2, 2, VOCAB_SIZE), -10.0)
    right[0, 0, 3] = right[0, 1, 7] = right[1, 1, 3] = 10.0
    return right


def test_loss() -> None:
    """The weight scales the unweighted loss."""
    plain = WordTerm(0.0, None)
    doubled = WordTerm(0.0, None, weight=2.0)
    once = plain.loss(nn.Identity(), -logits(), Y).item()
    assert doubled.loss(nn.Identity(), -logits(), Y).item() == pytest.approx(2 * once)


def test_balanced() -> None:
    y = torch.tensor([0, 0, 0, 0, 1, IGNORE])
    assert LabelTerm.balanced(y, 0.0) is None
    weight = LabelTerm.balanced(y, 0.5)
    assert weight is not None
    assert weight.shape == (VOCAB_SIZE,)
    assert torch.allclose(weight[:2], torch.tensor([2 / 3, 4 / 3]))
    assert weight[2:].eq(0).all()


def lexical() -> LexicalTerm:
    return LexicalTerm(1.0, Lexicon(torch.tensor([10, 20]), torch.zeros(2, 2)))


def test_label_term_labels() -> None:
    train = split()
    assert WordTerm(0.0, None).labels(train) is train.y


def test_auxiliary_term_labels() -> None:
    train = split()
    assert lexical().labels(train) is train.token


@pytest.mark.parametrize("objective", list(Objective))
def test_context_term_unweighted(objective: Objective) -> None:
    """The context term is its objective's loss over every span position."""
    context = ContextTerm(0.1, None, objective=objective)
    flat = objective.loss(logits().flatten(0, 1), Y.flatten(), 0.1)
    assert torch.equal(context.unweighted(nn.Identity(), logits(), Y), flat)
    assert context.unweighted(nn.Identity(), logits(), Y) < context.unweighted(nn.Identity(), -logits(), Y)


def test_word_term_unweighted() -> None:
    """The word term is the plain softmax over every span position."""
    word = WordTerm(0.1, None).unweighted(nn.Identity(), logits(), Y)
    expected = functional.cross_entropy(logits().flatten(0, 1), Y.flatten(), ignore_index=IGNORE, label_smoothing=0.1)
    assert torch.equal(word, expected)


def test_classified_unweighted() -> None:
    """Uniform logits over the classified targets cost log(count); an unclassifiable target is masked."""
    uniform = torch.zeros(1, 3, 2)
    loss = lexical().unweighted(nn.Identity(), uniform, torch.tensor([[10, 20, 99]]))
    assert loss.item() == pytest.approx(math.log(2))


def test_masked() -> None:
    """Uniform logits cost log(k) on each known row; unknown rows are neither counted nor averaged over."""
    uniform = torch.zeros(3, 5)
    assert Classified.masked(uniform, torch.tensor([0, 4, UNKNOWN])).item() == pytest.approx(math.log(5))
    sure = torch.full((1, 5), -50.0)
    sure[0, 4] = 50.0
    assert Classified.masked(sure, torch.tensor([4])).item() == pytest.approx(0.0, abs=1e-6)
    assert Classified.masked(uniform, torch.full((3,), UNKNOWN)).item() == 0.0

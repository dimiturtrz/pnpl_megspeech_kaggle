import math
from dataclasses import replace

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
from megspeech.objective.term import AuxiliaryTerm, Classified, ContextTerm, DistilledTerm, LabelTerm, WordTerm
from megspeech.objective.topk import TopKMix

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


def test_term_loss() -> None:
    """The weight scales the unweighted loss."""
    plain = WordTerm(0.0, None)
    doubled = WordTerm(0.0, None, weight=2.0)
    once = plain.loss(nn.Identity(), -logits(), Y).item()
    assert doubled.loss(nn.Identity(), -logits(), Y).item() == pytest.approx(2 * once)


def test_balanced() -> None:
    y = torch.tensor([0, 0, 0, 0, 1, IGNORE])
    uniform = torch.ones(len(y))
    assert LabelTerm.balanced(y, uniform, 0.0) is None
    weight = LabelTerm.balanced(y, uniform, 0.5)
    assert weight is not None
    assert weight.shape == (VOCAB_SIZE,)
    assert torch.allclose(weight[:2], torch.tensor([2 / 3, 4 / 3]))
    assert weight[2:].eq(0).all()
    drawn = LabelTerm.balanced(y, torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0, 1.0]), 0.5)
    assert drawn is not None
    assert torch.allclose(drawn[:2], torch.ones(2))


def lexical() -> LexicalTerm:
    return LexicalTerm(1.0, Lexicon(torch.tensor([10, 20]), torch.zeros(2, 2)))


def test_label_term_labels() -> None:
    train = split()
    assert WordTerm(0.0, None).labels(train) is train.y


def test_auxiliary_term_readout() -> None:
    assert lexical().readout(nn.Identity(), 4).in_features == 4


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


def distilled(share: float, taught: torch.Tensor) -> DistilledTerm:
    return DistilledTerm(0.0, None, taught=taught, y=torch.tensor([3, 7, IGNORE]), share=share)


def test_distilled_term_labels() -> None:
    term = distilled(0.5, torch.zeros(3, VOCAB_SIZE))
    with pytest.raises(ValueError, match="taught rows"):
        term.labels(split())
    train = split()
    sized = DistilledTerm(0.0, None, taught=torch.zeros(len(train.y), VOCAB_SIZE), y=train.y, share=0.5)
    assert torch.equal(sized.labels(train), torch.arange(len(train.y)))


def test_distilled_term_unweighted() -> None:
    """Share 0 is the label's loss alone; a teacher agreeing with the student adds no KL, one disagreeing adds some;
    rows that are padding or unlabelled count in neither."""
    rows = torch.tensor([[0, 1], [IGNORE, 2]])
    agreeing = logits()[[0, 0, 1], [0, 1, 1]].log_softmax(-1)
    labelled = ContextTerm(0.0, None).unweighted(nn.Identity(), logits(), torch.tensor([[3, 7], [IGNORE, IGNORE]]))
    assert torch.allclose(distilled(0.0, agreeing).unweighted(nn.Identity(), logits(), rows), labelled)
    assert torch.allclose(distilled(1.0, agreeing).unweighted(nn.Identity(), logits(), rows), torch.zeros(()))
    uniform = torch.full((3, VOCAB_SIZE), -math.log(VOCAB_SIZE))
    assert distilled(1.0, uniform).unweighted(nn.Identity(), logits(), rows) > 1.0
    shifted = distilled(1.0, agreeing - 2.0).unweighted(nn.Identity(), logits(), rows)
    assert torch.allclose(shifted, torch.zeros(()), atol=1e-6)


def test_distilled_term_temperature() -> None:
    """A teacher agreeing with the student costs nothing at any temperature; a disagreeing one's KL, scaled by the
    square of the temperature, changes with it."""
    rows = torch.tensor([[0, 1], [IGNORE, 2]])
    agreeing = logits()[[0, 0, 1], [0, 1, 1]]
    hot = DistilledTerm(0.0, None, taught=agreeing, y=torch.tensor([3, 7, IGNORE]), share=1.0, temperature=4.0)
    assert torch.allclose(hot.unweighted(nn.Identity(), logits(), rows), torch.zeros(()), atol=1e-6)
    uniform = torch.zeros(3, VOCAB_SIZE)
    cold = distilled(1.0, uniform).unweighted(nn.Identity(), logits(), rows)
    warm = DistilledTerm(0.0, None, taught=uniform, y=torch.tensor([3, 7, IGNORE]), share=1.0, temperature=4.0)
    assert not torch.allclose(warm.unweighted(nn.Identity(), logits(), rows), cold)


def test_distilled_term_unlabelled() -> None:
    """Only with `unlabelled` does a word without a label, where the teacher disagrees, add to the KL."""
    rows = torch.tensor([[0, 1], [IGNORE, 2]])
    taught = logits()[[0, 0, 1], [0, 1, 1]].log_softmax(-1)
    taught[2] = 0.0
    y = torch.tensor([3, 7, IGNORE])
    labelled = DistilledTerm(0.0, None, taught=taught, y=y, share=1.0)
    everywhere = DistilledTerm(0.0, None, taught=taught, y=y, share=1.0, unlabelled=True)
    assert torch.allclose(labelled.unweighted(nn.Identity(), logits(), rows), torch.zeros(()), atol=1e-6)
    assert everywhere.unweighted(nn.Identity(), logits(), rows) > 1.0


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
    assert Classified.masked(uniform, torch.tensor([0, 4, UNKNOWN]), torch.ones(3)).item() == pytest.approx(math.log(5))
    sure = torch.full((1, 5), -50.0)
    sure[0, 4] = 50.0
    assert Classified.masked(sure, torch.tensor([4]), torch.ones(1)).item() == pytest.approx(0.0, abs=1e-6)
    assert Classified.masked(uniform, torch.full((3,), UNKNOWN), torch.ones(3)).item() == 0.0


def test_averaged() -> None:
    """The weighted mean over the kept rows: weights of 1 are the plain mean, a heavier row pulls it, none kept is 0."""
    values = torch.tensor([0.0, 4.0, 100.0])
    keep = torch.tensor([True, True, False])
    assert AuxiliaryTerm.averaged(values, keep, torch.ones(3)).item() == pytest.approx(2.0)
    assert AuxiliaryTerm.averaged(values, keep, torch.tensor([3.0, 1.0, 9.0])).item() == pytest.approx(1.0)
    assert AuxiliaryTerm.averaged(values, torch.zeros(3, dtype=torch.bool), torch.ones(3)).item() == 0.0


def inside_logits() -> torch.Tensor:
    """Row 0 (token 10, class 0) classified right, row 1 (token 20, class 1) wrong."""
    logits = torch.tensor([[[10.0, -10.0], [10.0, -10.0]]])
    return logits


TOKEN_PAIR = torch.tensor([[10, 20]])
INSIDE = torch.tensor([[True, False]])


def test_weighted() -> None:
    """Rows at weight 1 are the unweighted loss; a heavier right row lowers the mean toward its zero loss."""
    term = lexical()
    flat = term.unweighted(nn.Identity(), inside_logits(), TOKEN_PAIR)
    assert torch.equal(term.weighted(nn.Identity(), inside_logits(), TOKEN_PAIR, torch.ones(1, 2)), flat)
    wrong = flat.item() * 2
    heavy = term.weighted(nn.Identity(), inside_logits(), TOKEN_PAIR, torch.tensor([[9.0, 1.0]]))
    assert heavy.item() == pytest.approx(wrong / 10)


def test_auxiliary_term_loss() -> None:
    """Weight 1 equals the old loss exactly; above 1 it moves toward the in-vocabulary rows; an all-OOV batch and a
    batch without `inside` read as weight 1."""
    plain = lexical()
    flat = plain.unweighted(nn.Identity(), inside_logits(), TOKEN_PAIR)
    assert torch.equal(plain.loss(nn.Identity(), inside_logits(), TOKEN_PAIR, INSIDE), flat)
    emphasised = replace(plain, vocab_weight=9.0)
    assert emphasised.loss(nn.Identity(), inside_logits(), TOKEN_PAIR, INSIDE).item() == pytest.approx(flat.item() / 5)
    assert emphasised.loss(nn.Identity(), inside_logits(), TOKEN_PAIR, torch.zeros(1, 2, dtype=torch.bool)) == flat
    assert torch.equal(emphasised.loss(nn.Identity(), inside_logits(), TOKEN_PAIR), flat)


def test_auxiliary_term_timed() -> None:
    assert not lexical().timed


def test_context_term_labels() -> None:
    """The context head's labels hide muted words; the word head's keep them."""
    train = split()
    muted = replace(train, muted=torch.tensor([True] + [False] * (len(train.y) - 1)))
    assert ContextTerm(0.0, None).labels(train).tolist() == train.y.tolist()
    assert ContextTerm(0.0, None).labels(muted).tolist() == [IGNORE, *train.y[1:].tolist()]
    assert WordTerm(0.0, None).labels(muted).tolist() == train.y.tolist()


def test_distilled_term_labels_muted() -> None:
    """A muted word is `IGNORE` for the label and the teacher alike, even with `unlabelled` matching."""
    train = split()
    muted = replace(train, muted=torch.tensor([True] + [False] * (len(train.y) - 1)))
    term = DistilledTerm(0.0, None, taught=torch.zeros(len(train.y), VOCAB_SIZE), y=train.y, share=0.5, unlabelled=True)
    assert term.labels(muted).tolist() == [IGNORE, *range(1, len(train.y))]


def test_context_term_hard() -> None:
    assert torch.equal(ContextTerm(0.0, None).hard(Y), Y)


def test_distilled_term_hard() -> None:
    term = distilled(0.5, torch.zeros(3, VOCAB_SIZE))
    assert term.hard(torch.tensor([[0, IGNORE]])).tolist() == [[int(term.y[0]), IGNORE]]


def test_auxiliary_term_unweighted() -> None:
    """Unweighted ignores `weight` and `vocab_weight`: it is the loss at unit rows."""
    term = replace(lexical(), vocab_weight=9.0)
    ones = torch.ones(1, 2)
    assert torch.equal(
        term.unweighted(nn.Identity(), inside_logits(), TOKEN_PAIR),
        term.weighted(nn.Identity(), inside_logits(), TOKEN_PAIR, ones),
    )
    assert torch.equal(
        replace(term, weight=3.0).unweighted(nn.Identity(), inside_logits(), TOKEN_PAIR),
        term.unweighted(nn.Identity(), inside_logits(), TOKEN_PAIR),
    )


def smoothed() -> torch.Tensor:
    return functional.cross_entropy(logits().flatten(0, 1), Y.flatten(), ignore_index=IGNORE, label_smoothing=0.1)


def test_context_term_labelled() -> None:
    """The head's own loss over flattened logits, before any top-k blend."""
    assert torch.allclose(ContextTerm(0.1, None).labelled(logits().flatten(0, 1), Y.flatten()), smoothed())


def test_word_term_labelled() -> None:
    assert torch.allclose(WordTerm(0.1, None).labelled(logits().flatten(0, 1), Y.flatten()), smoothed())


@pytest.mark.parametrize("term", [WordTerm(0.1, None), ContextTerm(0.1, None)])
def test_label_term_unweighted(term: LabelTerm) -> None:
    """Share 0 is the head's own loss exactly; share 1 is the top-k term alone, ~0 when every word ranks first."""
    plain = term.unweighted(nn.Identity(), logits(), Y)
    off = replace(term, topk=TopKMix(share=0.0, temperature=0.5)).unweighted(nn.Identity(), logits(), Y)
    assert torch.equal(off, plain)
    full = replace(term, topk=TopKMix(share=1.0)).unweighted(nn.Identity(), logits(), Y)
    assert full < 1e-6
    assert replace(term, topk=TopKMix(share=1.0)).unweighted(nn.Identity(), -logits(), Y) > 1.0

import math

import pytest
import torch

from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.t5 import Lexicon

TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


def term() -> LexicalTerm:
    return LexicalTerm(3.0, Lexicon(torch.tensor([10, 20]), torch.zeros(2, DIM)))


def test_head() -> None:
    assert term().head(16)(torch.zeros(2, 16)).shape == (2, 2)


def test_count() -> None:
    assert term().count() == 2


def test_classify() -> None:
    assert term().classify(torch.tensor([[20, 10, 99]])).tolist() == [[1, 0, -1]]


def test_unweighted() -> None:
    """Tokens 10 and 20 are the lexicon's 2 words; uniform logits cost log 2 per known row, unknown rows nothing."""
    lexical = term()
    head = lexical.head(16)
    assert lexical.loss(lexical.read(head, torch.zeros(2, 3, 2), TOKENS)).item() == pytest.approx(3.0 * math.log(2))
    assert lexical.unweighted(lexical.read(head, torch.zeros(2, 3, 2), torch.full((2, 3), 99))).item() == 0.0
    sure = torch.tensor([[[50.0, -50.0], [-50.0, 50.0], [0.0, 0.0]]])
    assert lexical.unweighted(lexical.read(head, sure, TOKENS[:1])).item() == pytest.approx(0.0, abs=1e-6)

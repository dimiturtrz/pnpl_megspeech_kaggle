import math

import pytest
import torch

from megspeech.data.labels.syntax import TAGS, PartsOfSpeech
from megspeech.objective.auxiliary.syntactic import SyntacticTerm

NOUN, VERB = TAGS.index("NOUN"), TAGS.index("VERB")
TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


def term() -> SyntacticTerm:
    return SyntacticTerm(2.0, PartsOfSpeech(torch.tensor([10, 20]), torch.tensor([NOUN, VERB])))


def test_head() -> None:
    assert term().head(16)(torch.zeros(2, 16)).shape == (2, len(TAGS))


def test_count() -> None:
    assert term().count() == len(TAGS)


def test_classify() -> None:
    assert term().classify(torch.tensor([[20, 10, 99]])).tolist() == [[VERB, NOUN, -1]]


def test_unweighted() -> None:
    """Uniform logits cost log 17 per tagged row; untagged rows cost nothing."""
    syntactic = term()
    head = syntactic.head(16)
    uniform = torch.zeros(2, 3, len(TAGS))
    assert syntactic.loss(syntactic.read(head, uniform, TOKENS)).item() == pytest.approx(2.0 * math.log(len(TAGS)))
    assert syntactic.unweighted(syntactic.read(head, uniform, torch.full((2, 3), 99))).item() == 0.0

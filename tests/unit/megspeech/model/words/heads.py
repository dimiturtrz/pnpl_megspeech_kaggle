import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.heads import LinearHead, WordHead

WIDTH = 6


def test_linear_head() -> None:
    assert LinearHead(WIDTH, 0.5)[1].weight.shape == (VOCAB_SIZE, WIDTH)


@pytest.mark.parametrize("kind", list(WordHead))
def test_build(kind: WordHead) -> None:
    assert kind.build(WIDTH, 0.0) is not None


@pytest.mark.parametrize("kind", list(WordHead))
def test_built_head_reads(kind: WordHead) -> None:
    head = kind.build(WIDTH, 0.0)
    head.attach(WIDTH)
    assert head.read(torch.zeros(2, WIDTH), torch.zeros(2, WIDTH)).shape == (2, VOCAB_SIZE)

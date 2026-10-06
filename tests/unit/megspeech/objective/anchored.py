import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.objective.anchored import INITIAL_SCALE, AnchoredHead

GENERATOR = torch.Generator().manual_seed(0)
WIDTH = 8
DIM = 5


def test_forward() -> None:
    anchors = torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR)
    head = AnchoredHead(WIDTH, anchors)
    logits = head.forward(torch.randn(2, 3, WIDTH, generator=GENERATOR))
    assert logits.shape == (2, 3, VOCAB_SIZE)
    assert logits.abs().max() <= INITIAL_SCALE + 1e-4
    assert head.scale.exp().item() == pytest.approx(INITIAL_SCALE)
    assert not any(p is head.anchors for p in head.parameters())
    assert torch.allclose(head.anchors, anchors)


def test_forward_trainable() -> None:
    head = AnchoredHead(WIDTH, torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR), trainable=True)
    head.forward(torch.randn(1, WIDTH, generator=GENERATOR)).sum().backward()
    assert isinstance(head.anchors, torch.nn.Parameter)
    assert head.anchors.grad is not None

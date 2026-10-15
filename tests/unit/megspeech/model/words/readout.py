import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.heads import LinearHead

WIDTH = 6


def test_attach() -> None:
    head = LinearHead(WIDTH, 0.0)
    before = [p.clone() for p in head.parameters()]
    head.attach(WIDTH + 1)
    assert all(torch.equal(a, b) for a, b in zip(before, head.parameters(), strict=True))


def test_read() -> None:
    head = LinearHead(WIDTH, 0.0)
    slots = torch.randn(2, 3, WIDTH, generator=torch.Generator().manual_seed(0))
    assert torch.equal(head.read(slots, torch.zeros(2, 3, 1)), head(slots))
    assert head.read(slots, torch.zeros(2, 3, 1)).shape == (2, 3, VOCAB_SIZE)

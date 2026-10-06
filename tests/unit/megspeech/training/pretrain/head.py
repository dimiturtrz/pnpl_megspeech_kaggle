import torch

from megspeech.training.pretrain.head import Head
from megspeech.training.pretrain.prototypes import Prototypes

CONFIG = Prototypes(k=8, bottleneck=6, hidden=10)


def test_forward() -> None:
    torch.manual_seed(0)
    head = Head(5, CONFIG)
    logits = head.forward(torch.randn(7, 5))
    assert logits.shape == (7, 8)
    assert logits.abs().max() <= 1 + 1e-5
    assert head.prototypes.shape == (8, 6)

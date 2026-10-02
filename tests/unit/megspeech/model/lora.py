import torch
from torch import nn

from megspeech.model.lora import Lora

RANK = 2


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)
        self.other = nn.Linear(4, 4)


def block() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(Attention(), nn.Sequential(Attention()))


def test_forward():
    base = nn.Linear(4, 3)
    lora = Lora(base, RANK)
    x = torch.randn(5, 4)
    assert torch.equal(lora.forward(x), base(x))
    nn.init.ones_(lora.up.weight)
    assert not torch.allclose(lora.forward(x), base(x))


def test_adapter():
    lora = Lora(nn.Linear(4, 3), RANK)
    assert [p.shape for p in lora.adapter()] == [(RANK, 4), (3, RANK)]
    assert not any(p.requires_grad for p in lora.base.parameters())


def test_inject():
    module = block()
    assert Lora.inject(module, 0) == 0
    assert Lora.inject(module, RANK) == 4
    attention = module[1][0]
    assert isinstance(attention.qkv, Lora)
    assert isinstance(attention.proj, Lora)
    assert isinstance(attention.other, nn.Linear)

import torch
from torch import nn

from megspeech.training.hook import Scheduled


def scheduled() -> Scheduled:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.5)
    return Scheduled(torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1)))


def test_lr() -> None:
    assert scheduled().lr() == 0.5


def test_step() -> None:
    hook = scheduled()
    hook.step(nn.Identity(), hook.lr())
    assert hook.lr() == 0.25

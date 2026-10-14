import torch
from torch import nn

from megspeech.training.regularise.sharpness import Flat, Sharp

RHO = 0.5


def weights() -> nn.Parameter:
    return nn.Parameter(torch.tensor([3.0, 4.0]))


def bowl(w: nn.Parameter) -> torch.Tensor:
    """Half the squared norm: its gradient is the weights themselves."""
    return w.square().sum() / 2


def test_of() -> None:
    assert type(Sharp.of(0.0)) is Flat
    assert Sharp.of(RHO) == Sharp(RHO)


def test_backward() -> None:
    """Plain descent leaves the gradient at the weights; SAM the one `rho` up the normalised gradient, and puts the
    weights back."""
    w = weights()
    Flat().backward(bowl(w), lambda: bowl(w), (w,))
    assert torch.equal(w.grad, torch.tensor([3.0, 4.0]))
    w = weights()
    Sharp(RHO).backward(bowl(w), lambda: bowl(w), (w,))
    assert torch.allclose(w.grad, torch.tensor([3.0, 4.0]) * (1 + RHO / 5))
    assert torch.equal(w.detach(), torch.tensor([3.0, 4.0]))

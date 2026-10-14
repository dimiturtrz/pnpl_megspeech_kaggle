import torch
from torch import nn

from megspeech.training.regularise.sharpness import ETA, Adaptive, Flat, Sharp

RHO = 0.5


def weights() -> nn.Parameter:
    return nn.Parameter(torch.tensor([3.0, 4.0]))


def bowl(w: nn.Parameter) -> torch.Tensor:
    """Half the squared norm: its gradient is the weights themselves."""
    return w.square().sum() / 2


def grad(w: nn.Parameter) -> torch.Tensor:
    assert w.grad is not None
    return w.grad


def test_of() -> None:
    assert type(Sharp.of(0.0)) is Flat
    assert Sharp.of(RHO) == Sharp(RHO)
    assert Sharp.of(RHO, adaptive=True) == Adaptive(RHO)
    assert type(Sharp.of(0.0, adaptive=True)) is Flat


def test_sharp_stretch() -> None:
    g = torch.tensor([1.0, 2.0])
    assert torch.equal(Sharp(RHO).stretch(weights(), g), g)


def test_adaptive_stretch() -> None:
    g = torch.tensor([1.0, 2.0])
    assert torch.allclose(Adaptive(RHO).stretch(weights(), g), torch.tensor([3.0 + ETA, 2 * (4.0 + ETA)]))


def test_flat_backward() -> None:
    """Plain descent leaves the gradient at the weights themselves."""
    w = weights()
    Flat().backward(bowl(w), lambda: bowl(w), (w,))
    assert torch.equal(grad(w), torch.tensor([3.0, 4.0]))


def test_sharp_backward() -> None:
    """SAM leaves the gradient `rho` up the normalised gradient, and puts the weights back."""
    w = weights()
    Sharp(RHO).backward(bowl(w), lambda: bowl(w), (w,))
    assert torch.allclose(grad(w), torch.tensor([3.0, 4.0]) * (1 + RHO / 5))
    assert torch.equal(w.detach(), torch.tensor([3.0, 4.0]))


def test_adaptive_backward() -> None:
    """ASAM on the bowl: the ascent T^2 g / |T g| with T = |w| + eta, so the gradient there is w plus that step."""
    w = weights()
    Adaptive(RHO).backward(bowl(w), lambda: bowl(w), (w,))
    t = torch.tensor([3.0, 4.0]) + ETA
    g = torch.tensor([3.0, 4.0])
    step = RHO * t * t * g / torch.linalg.vector_norm(t * g)
    assert torch.allclose(grad(w), g + step)
    assert torch.equal(w.detach(), torch.tensor([3.0, 4.0]))

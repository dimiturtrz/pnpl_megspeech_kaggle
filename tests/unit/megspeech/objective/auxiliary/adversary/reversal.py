import torch

from megspeech.objective.auxiliary.adversary.reversal import Reversal


def test_forward() -> None:
    """The value passes through; the gradient is scaled by minus the strength."""
    reversal = Reversal()
    reversal.strength.fill_(0.5)
    x = torch.ones(1, 2, 3, requires_grad=True)
    y = reversal.forward(x)
    assert torch.equal(y, x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.full_like(x, -0.5))

import torch

from megspeech.model.ssm import Ssm

GENERATOR = torch.Generator().manual_seed(0)


def test_kernel() -> None:
    ssm = Ssm(4, 3, 0.0)
    kernel = ssm.kernel(10)
    assert kernel.shape == (2, 4, 10)
    with torch.no_grad():
        ssm.output.zero_()
    assert torch.all(ssm.kernel(10) == 0)


def test_kernel_decays() -> None:
    ssm = Ssm(4, 3, 0.0)
    with torch.no_grad():
        ssm.output.fill_(1.0)
    kernel = ssm.kernel(10)
    assert torch.all(kernel[..., 1:] <= kernel[..., :-1])


def test_forward() -> None:
    """An impulse mid-window reaches both window edges: one mode set runs forward in time, the other backward."""
    ssm = Ssm(4, 3, 0.0).eval()
    x = torch.zeros(2, 4, 10)
    x[..., 5] = 1.0
    with torch.no_grad():
        ssm.skip.zero_()
        mixed = ssm.forward(x) - x
    assert mixed.shape == x.shape
    assert torch.all(mixed[..., 0].abs().sum() > 0)
    assert torch.all(mixed[..., -1].abs().sum() > 0)


def test_forward_valid() -> None:
    ssm = Ssm(4, 3, 0.0).train()
    x = torch.randn(3, 4, 10, generator=GENERATOR)
    valid = torch.tensor([True, True, False])
    assert torch.allclose(ssm.forward(x, valid)[:2], ssm.forward(x[:2]), atol=1e-5)

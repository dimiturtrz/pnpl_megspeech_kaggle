import pytest
import torch

from megspeech.model.words.layers.ssm import Ssm

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuFFT rejects bfloat16; CPU FFT accepts it")
def test_forward_under_autocast() -> None:
    ssm = Ssm(4, 3, 0.0).eval().cuda()
    x = torch.randn(2, 4, 10, generator=GENERATOR).to("cuda", torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mixed = ssm.forward(x)
    assert mixed.dtype == torch.bfloat16
    assert torch.isfinite(mixed).all()

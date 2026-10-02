import torch
from torch import nn

from megspeech.norm import MaskedBatchNorm1d

GENERATOR = torch.Generator().manual_seed(0)
VALID = torch.tensor([True, True, False, True, False])


def pair() -> tuple[MaskedBatchNorm1d, nn.BatchNorm1d, torch.Tensor]:
    return MaskedBatchNorm1d(4), nn.BatchNorm1d(4), torch.randn(5, 4, 7, generator=GENERATOR)


def test_forward() -> None:
    masked, plain, x = pair()
    assert torch.allclose(masked.forward(x, VALID)[VALID], plain(x[VALID]), atol=1e-5)
    assert torch.allclose(masked.forward(x), plain(x), atol=1e-5)
    assert torch.allclose(masked.eval().forward(x, VALID), plain.eval()(x), atol=1e-5)


def test_track() -> None:
    masked, plain, x = pair()
    masked.forward(x, VALID)
    plain(x[VALID])
    for name, buffer in plain.named_buffers():
        assert torch.allclose(masked.get_buffer(name).float(), buffer.float(), atol=1e-5), name
    fresh = MaskedBatchNorm1d(4)
    fresh.track(torch.ones(4), torch.ones(4), torch.tensor(2.0))
    assert torch.allclose(fresh.running_mean, torch.full((4,), 0.1))
    assert torch.allclose(fresh.running_var, torch.full((4,), 1.1))

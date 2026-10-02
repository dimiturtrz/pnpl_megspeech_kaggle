import torch

from megspeech.training.augment import Augment


def test_call() -> None:
    torch.manual_seed(0)
    x = torch.ones(64, 30, 5)
    assert torch.equal(Augment()(x), x)
    dropped = Augment(dropout=0.5)(x)
    assert 0.3 < (dropped[..., 0] == 0).float().mean() < 0.7
    assert torch.all(dropped.std(dim=-1) == 0)
    scaled = Augment(gain=0.2)(x)[..., 0]
    assert torch.all(scaled > 0)
    assert 0.1 < scaled.log().std() < 0.3
    assert Augment(jitter=2).jitter == 2


def test_call_mask_zeroes_one_span_per_window_on_every_sensor() -> None:
    torch.manual_seed(0)
    masked = Augment(mask=3)(torch.ones(64, 4, 10))
    zeros = masked == 0
    assert torch.all(zeros.sum(dim=-1) == 3)
    assert torch.all(zeros == zeros[:, :1])
    first = zeros.int().argmax(dim=-1)
    assert torch.all(zeros.gather(-1, first[..., None] + torch.arange(3)))


def test_call_band_removes_one_band_of_bins() -> None:
    torch.manual_seed(0)
    t = torch.arange(20.0)
    x = torch.stack([torch.cos(2 * torch.pi * k * t / 20) for k in range(11)]).sum(0).expand(8, 2, 20)
    kept = torch.fft.rfft(Augment(band=4)(x), dim=-1).abs() > 1
    assert torch.all(kept.sum(dim=-1) == 11 - 4)


def test_call_noise_scales_with_spread_and_spares_padding() -> None:
    torch.manual_seed(0)
    x = torch.randn(256, 4, 50) * torch.tensor([1.0, 10.0, 1.0, 1.0])[:, None]
    added = (Augment(noise=0.5)(x) - x).std(dim=(0, 2))
    assert torch.allclose(added / x.std(dim=(0, 2)), torch.full((4,), 0.5), atol=0.05)
    x[..., 40:] = 0
    assert torch.all(Augment(noise=0.5)(x)[..., 40:] == 0)


def test_masked() -> None:
    x = torch.ones(2, 1, 4)
    assert torch.equal(Augment.masked(x, 0, 4), x)
    assert torch.all(Augment.masked(x, 4, 4) == 0)

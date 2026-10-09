import pytest
import torch

from megspeech.training.regularise.augment import Augment


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


def test_call_half_precision_near_overflow_stays_finite() -> None:
    torch.manual_seed(0)
    x = torch.full((64, 4, 50), 60000.0, dtype=torch.float16)
    x[..., ::2] = -60000.0
    perturbed = Augment(gain=0.5, band=4, noise=0.5)(x)
    assert perturbed.dtype == torch.float32
    assert torch.isfinite(perturbed).all()


def test_call_chance_gates_time_and_band_masks_per_window() -> None:
    torch.manual_seed(0)
    x = torch.ones(64, 4, 10)
    assert torch.all((Augment(mask=3, chance=1.0)(x) == 0).sum(dim=-1) == 3)
    assert torch.equal(Augment(mask=3, band=2, chance=0.0)(x), x)
    hit = (Augment(mask=3, chance=0.5)(x) == 0).any(dim=(1, 2))
    assert 0 < hit.sum() < 64
    spectrum = torch.fft.rfft(Augment(band=2, chance=0.5)(torch.randn(64, 4, 10)), dim=-1).abs() < 1e-5
    assert 0 < spectrum.any(dim=(1, 2)).sum() < 64


def test_post_init_rejects_chance_outside_unit_interval() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="chance"):
            Augment(chance=bad)


def test_masked() -> None:
    x = torch.ones(2, 1, 4)
    assert torch.equal(Augment.masked(x, 0, 4), x)
    assert torch.all(Augment.masked(x, 4, 4) == 0)
    assert torch.equal(Augment.masked(x, 4, 4, 0.0), x)


def test_mixed() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 3, 20)
    assert torch.equal(Augment().mixed(x), x)
    added = Augment(mix=0.5).mixed(x) - x
    assert 0 < added.abs().mean() < x.abs().mean()
    x[..., 15:] = 0
    assert torch.all(Augment(mix=0.5).mixed(x)[..., 15:] == 0)
    assert torch.equal(Augment(mix=0.5)(torch.zeros(4, 2, 5)), torch.zeros(4, 2, 5))


def test_sensors() -> None:
    x = torch.randn(4, 3, 5)
    state = torch.get_rng_state()
    assert Augment().sensors(x) is x
    assert torch.equal(torch.get_rng_state(), state)
    assert Augment(gain=0.2).sensors(x) is not x


def test_scaled() -> None:
    x = torch.ones(8, 3, 5)
    assert torch.all(Augment(dropout=1.0).scaled(x) == 0)
    assert torch.equal(Augment().scaled(x), x)


def test_unscaled() -> None:
    x = torch.randn(2, 2, 2)
    assert Augment.unscaled(x) is x


def unit_scaled(x: torch.Tensor, dropout: float, gain: float) -> torch.Tensor:
    shape = (*x.shape[:2], 1)
    keep = torch.rand(shape, device=x.device) >= dropout
    return x * (keep * torch.exp(gain * torch.randn(shape, device=x.device)))


def test_sensors_matches_unit_scaling() -> None:
    x = (torch.randn(16, 30, 75) * 1e3).half().float()
    torch.manual_seed(1)
    assert torch.equal(Augment().sensors(x), unit_scaled(x, 0.0, 0.0))
    torch.manual_seed(1)
    expected = unit_scaled(x, 0.2, 0.1)
    torch.manual_seed(1)
    assert torch.equal(Augment(dropout=0.2, gain=0.1).sensors(x), expected)

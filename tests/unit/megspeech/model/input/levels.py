import torch

from megspeech.model.input.levels import KNOWN, Levels
from megspeech.model.input.scaler import Scaler

CHANNELS = 2
SAMPLES = 10
WORDS = 200


def heard() -> tuple[torch.Tensor, torch.Tensor]:
    """Subject 0's windows at unit scale, subject 1's at ten times the gain and offset by 3."""
    x = torch.randn(2 * WORDS, CHANNELS, SAMPLES, generator=torch.Generator().manual_seed(0))
    subject = torch.arange(2).repeat_interleave(WORDS)
    x[WORDS:] = 10 * x[WORDS:] + 3
    return x, subject


def fitted() -> tuple[Levels, Scaler, torch.Tensor, torch.Tensor]:
    x, subject = heard()
    scaler = Scaler(x)
    levels = Levels(CHANNELS, active=True)
    levels.fit(lambda rows: x[rows], subject, scaler)
    return levels, scaler, x, subject


def test_fit() -> None:
    levels, scaler, x, subject = fitted()
    for who in (0, 1):
        own = subject == who
        median, spread = Scaler.quartiles(scaler(levels(x[own], subject[own])))
        assert torch.allclose(median, torch.zeros_like(median), atol=1e-5)
        assert torch.allclose(spread, torch.ones_like(spread), atol=1e-5)
    assert torch.equal(levels.gain[2], torch.ones(CHANNELS, 1))


def test_fit_inactive() -> None:
    x, subject = heard()
    levels = Levels(CHANNELS)
    levels.fit(lambda rows: x[rows], subject, Scaler(x))
    assert torch.equal(levels.gain, torch.ones(KNOWN + 1, CHANNELS, 1))
    assert not levels.state_dict()


def test_fit_flat_channel() -> None:
    x, subject = heard()
    x[WORDS:, 1] = 7.0
    scaler = Scaler(x)
    levels = Levels(CHANNELS, active=True)
    levels.fit(lambda rows: x[rows], subject, scaler)
    assert levels.gain[1, 1].item() == 0.0
    assert torch.equal(levels(x[-1:], subject[-1:])[:, 1], scaler.median[1].expand(1, SAMPLES))


def test_forward() -> None:
    levels, _, x, subject = fitted()
    assert torch.equal(Levels(CHANNELS).forward(x, subject), x)
    unseen = torch.tensor([KNOWN, KNOWN + 6, torch.iinfo(torch.long).max])
    assert torch.equal(levels.forward(x[:3], unseen), x[:3])
    assert not torch.equal(levels.forward(x[-1:], subject[-1:]), x[-1:])


def test_mapped() -> None:
    levels = Levels(CHANNELS, active=True, subjects=1)
    levels.gain[0] = 2.0
    levels.offset[0] = -1.0
    x = torch.ones(1, CHANNELS, SAMPLES)
    assert torch.equal(levels.mapped(x, torch.tensor([0])), torch.ones_like(x))
    assert torch.equal(levels.mapped(x, torch.tensor([1])), x)

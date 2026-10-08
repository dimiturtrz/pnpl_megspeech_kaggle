import math
from typing import override

import torch
from torch import nn

from megspeech.data.batching.split import SpanInput
from megspeech.model.input.lowpass import LowPass

RATE = 50.0
SAMPLES = 50
TIME = torch.arange(SAMPLES) / RATE


def tone(hertz: float) -> torch.Tensor:
    return torch.sin(2 * math.pi * hertz * TIME).expand(2, 3, SAMPLES)


class Seen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = torch.empty(0)

    @override
    def forward(self, spans: SpanInput) -> tuple[torch.Tensor, torch.Tensor]:
        self.x = spans.x
        return spans.onset, spans.onset


def test_filtered() -> None:
    low, high = tone(3.0), tone(20.0)
    passed = LowPass(Seen(), RATE, 12.0).filtered(low + high)
    assert torch.allclose(passed, low, atol=1e-4)
    assert torch.allclose(LowPass(Seen(), RATE, 25.0).filtered(low + high), low + high, atol=1e-4)
    assert torch.equal(LowPass(Seen(), RATE, 12.0).filtered(torch.zeros(2, 3, SAMPLES)), torch.zeros(2, 3, SAMPLES))


def test_forward() -> None:
    seen = Seen()
    onset = torch.zeros(1, 2)
    out = LowPass(seen, RATE, 12.0).forward(
        SpanInput(tone(20.0), onset, torch.zeros(1, 2, dtype=torch.long), torch.zeros(2))
    )
    assert out[0] is onset
    assert torch.allclose(seen.x, torch.zeros(2, 3, SAMPLES), atol=1e-4)

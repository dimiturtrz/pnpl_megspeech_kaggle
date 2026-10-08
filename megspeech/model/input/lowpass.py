"""A model reading low-passed windows: an instrument for how much of its score rides on the top of its band."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput


class LowPass(nn.Module):
    """`model` reading windows sampled at `rate` with every frequency above `cutoff` removed."""

    def __init__(self, model: nn.Module, rate: float, cutoff: float) -> None:
        super().__init__()
        self.model = model
        self.rate = rate
        self.cutoff = cutoff

    def filtered(self, x: Float[Tensor, "r c t"]) -> Float[Tensor, "r c t"]:
        """`x` without its frequencies above the cutoff, as float32."""
        length = x.shape[-1]
        kept = torch.fft.rfftfreq(length, d=1 / self.rate, device=x.device) <= self.cutoff
        return torch.fft.irfft(torch.fft.rfft(x.float(), dim=-1) * kept, n=length, dim=-1)

    @override
    def forward(self, spans: SpanInput) -> tuple[Float[Tensor, "g w v"], ...]:
        return self.model(spans.reading(self.filtered(spans.x)))

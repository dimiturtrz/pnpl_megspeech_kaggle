"""One word encoder trained apart, read inside another's window: its stretch of that window, its scaler and its CNN,
as its own run read them (megspeech-7gf.90)."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.input.scaler import Scaler
from megspeech.model.words.conv import ConvConfig, ConvWords


@dataclass(frozen=True)
class Crop:
    """`length` samples from `start` of a window, centred on their own valid samples as a window of that stretch is."""

    start: int
    length: int

    @classmethod
    def many(cls, starts: tuple[int, ...], lengths: tuple[int, ...]) -> tuple["Crop", ...]:
        """One crop per start, `lengths` running alongside."""
        return tuple(cls(start, length) for start, length in zip(starts, lengths, strict=True))

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c s"]:
        cut = x[..., self.start : self.start + self.length]
        present = cut.ne(0).any(dim=1, keepdim=True)
        mean = (cut * present).sum(-1, keepdim=True) / present.sum(-1, keepdim=True).clamp(min=1)
        return (cut - mean) * present


class Expert(nn.Module):
    """One expert's word embedding: its crop, its own scaler and its CNN."""

    def __init__(self, crop: Crop, config: ConvConfig, channels: int) -> None:
        super().__init__()
        self.crop = crop
        self.scale = Scaler(torch.zeros(1, channels, 1))
        self.words = ConvWords(config, channels)

    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        cut = self.crop(x)
        scaled = self.words.normalise(self.scale(cut), cut.ne(0).any(dim=1, keepdim=True))
        return self.words.embed(scaled, valid)

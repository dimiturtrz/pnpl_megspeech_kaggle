"""What a word is told of when it was heard: its place in the span, and the gaps to its neighbours' onsets."""

from dataclasses import dataclass
from enum import StrEnum
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE

KAGGLE_GAP = 0.21
"""Median seconds between consecutive vocabulary words in a Kaggle test sentence."""

SHORTEST = 0.02
"""One 50 Hz sample: two onsets never fall closer."""


class Clock(StrEnum):
    """What the transformer is told of each word's position: its onset, or only its rank in the span."""

    ONSET = "onset"
    ORDER = "order"

    def times(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w"]:
        """Seconds to encode; word order is laid out at the Kaggle test's typical spacing so the periods still fit."""
        return onset if self is Clock.ONSET else (mask.cumsum(-1) - 1).to(onset.dtype) * KAGGLE_GAP


@dataclass(frozen=True)
class Jitter:
    """Gaussian error of `sigma` seconds on every word onset, as training noise on what the timing input is told: the
    labelled onsets are 10 ms-rounded alignments, and an alignment off by a sample or two must not move the score
    (megspeech-7gf.110). 0 leaves the onsets as they are."""

    sigma: float = 0.0

    def apply(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w"]:
        """`onset` with every word of `mask` moved, still in order and zero at each span's first word."""
        if self.sigma == 0.0:
            return onset
        return self.ordered(onset + self.sigma * torch.randn_like(onset), mask, onset)

    @staticmethod
    def ordered(
        moved: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], onset: Float[Tensor, "g w"]
    ) -> Float[Tensor, "g w"]:
        """`moved` re-sorted among the words of `mask`, padding kept as in `onset`, and re-zeroed at the first word."""
        ordered = moved.where(mask, torch.full_like(moved, torch.inf)).sort(dim=1).values
        return (ordered - ordered[:, :1] * mask[:, :1]).where(mask, onset)


FEATURES = 14
"""`TimingHead.features` per word: four neighbours' log gaps and whether each exists, place, length, place over the
last place, places to the end, the span's log rate, and whether the word is alone."""

HIDDEN = 256
"""Width of the timing head's two hidden layers: a 2x256 MLP on these features alone scored 0.55 novel BAcc@10
(megspeech-7gf.77), the context model 0.66."""

NEIGHBOURS = (-2, -1, 1, 2)


class TimingHead(nn.Module):
    """Logits over the vocabulary from when each word was heard alone, added to the context head's: a product of
    experts trained jointly (Clark et al. 2019, keeping the timing expert rather than discarding it).

    With the transformer told no gaps (`Intervals.NONE`), the MEG path is left only what timing cannot explain, where
    a context model given the gaps learns the timing cue itself (megspeech-7gf.77). Inactive, it adds nothing and
    saves nothing, so models trained before it load."""

    def __init__(self, vocabulary: int, *, active: bool = True) -> None:
        super().__init__()
        self.active = active
        layers = (nn.Linear(FEATURES, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU())
        self.layers = nn.Sequential(*layers, nn.Linear(HIDDEN, vocabulary)) if active else nn.Sequential()

    @classmethod
    def over_vocabulary(cls, *, active: bool = True) -> "TimingHead":
        """A head whose logits cover the whole vocabulary."""
        return cls(VOCAB_SIZE, active=active)

    @staticmethod
    def neighbour(onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], step: int) -> Float[Tensor, "g w 2"]:
        """Log gap to the word `step` places on, in Kaggle test gaps, beside whether that word exists."""
        place = torch.arange(onset.shape[1], device=onset.device) + step
        inside = (place >= 0) & (place < onset.shape[1])
        known = mask & torch.roll(mask, -step, dims=1) & inside
        gap = (torch.roll(onset, -step, dims=1) - onset).abs().clamp(min=SHORTEST) / KAGGLE_GAP
        return torch.stack([gap.log().where(known, 0.0), known.to(onset.dtype)], dim=-1)

    @staticmethod
    def features(onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w f"]:
        """Each word's gaps, place in its span and the span's rate (`FEATURES`); zeros at padded positions."""
        gaps = [TimingHead.neighbour(onset, mask, step) for step in NEIGHBOURS]
        place = (mask.cumsum(-1) - 1).clamp(min=0).to(onset.dtype)
        length = mask.sum(-1, keepdim=True).to(onset.dtype).expand_as(onset)
        last = (length - 1).clamp(min=1)
        duration = onset.where(mask, 0.0).amax(-1, keepdim=True).expand_as(onset)
        rate = (duration / last).clamp(min=SHORTEST).div(KAGGLE_GAP).log()
        span = [place.log1p(), length.log1p(), place / last, (length - 1 - place).clamp(min=0).log1p(), rate]
        alone = length.eq(1).to(onset.dtype)
        return torch.cat([*gaps, torch.stack([*span, alone], dim=-1)], dim=-1) * mask[..., None]

    @override
    def forward(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "..."]:
        """Each position's timing logits; a scalar 0 when inactive."""
        return self.layers(self.features(onset, mask)) if self.active else onset.new_zeros(())

    def frozen(self, state: dict[str, Tensor]) -> None:
        """Hold the weights of a timing expert trained alone (a `TimingOnly`), which no optimiser step then moves."""
        self.load_state_dict(state)
        self.requires_grad_(requires_grad=False)

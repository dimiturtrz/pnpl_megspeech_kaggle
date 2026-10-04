"""What a word is told of when it was heard: its place in the span, and the gaps to its neighbours' onsets."""

from enum import StrEnum
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

HOLDOUT_GAP = 0.21
"""Median seconds between consecutive vocabulary words in a holdout sentence."""

SHORTEST = 0.02
"""One 50 Hz sample: two onsets never fall closer."""


class Clock(StrEnum):
    """What the transformer is told of each word's position: its onset, or only its rank in the span."""

    ONSET = "onset"
    ORDER = "order"

    def times(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w"]:
        """Seconds to encode; word order is laid out at the holdout's typical spacing so the periods still fit."""
        return onset if self is Clock.ONSET else (mask.cumsum(-1) - 1).to(onset.dtype) * HOLDOUT_GAP


class Intervals(StrEnum):
    """Whether each word is told its gaps to its span neighbours' onsets.

    `log`: the log gaps to the next and previous word's onsets, in holdout gaps, each beside whether that neighbour
    exists. A word's duration identifies it far above chance (arXiv 2609.40359: log inter-onset time alone 22.9%
    top-1 of 50); a span's last word has no next onset, in training and holdout alike."""

    NONE = "none"
    LOG = "log"

    @property
    def width(self) -> int:
        return 4 if self is Intervals.LOG else 0

    def features(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w k"]:
        if self is Intervals.NONE:
            return onset.new_zeros(*onset.shape, 0)
        pair = mask[:, 1:] & mask[:, :-1]
        gap = torch.where(pair, (onset[:, 1:] - onset[:, :-1]).clamp(min=SHORTEST) / HOLDOUT_GAP, 1.0).log()
        pad = nn.functional.pad
        known = pair.to(onset.dtype)
        return torch.stack([pad(gap, (0, 1)), pad(known, (0, 1)), pad(gap, (1, 0)), pad(known, (1, 0))], dim=-1)


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

    @staticmethod
    def neighbour(onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], step: int) -> Float[Tensor, "g w 2"]:
        """Log gap to the word `step` places on, in holdout gaps, beside whether that word exists."""
        place = torch.arange(onset.shape[1], device=onset.device) + step
        inside = (place >= 0) & (place < onset.shape[1])
        known = mask & torch.roll(mask, -step, dims=1) & inside
        gap = (torch.roll(onset, -step, dims=1) - onset).abs().clamp(min=SHORTEST) / HOLDOUT_GAP
        return torch.stack([gap.log().where(known, 0.0), known.to(onset.dtype)], dim=-1)

    @staticmethod
    def features(onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w f"]:
        """Each word's gaps, place in its span and the span's rate (`FEATURES`); zeros at padded positions."""
        gaps = [TimingHead.neighbour(onset, mask, step) for step in NEIGHBOURS]
        place = (mask.cumsum(-1) - 1).clamp(min=0).to(onset.dtype)
        length = mask.sum(-1, keepdim=True).to(onset.dtype).expand_as(onset)
        last = (length - 1).clamp(min=1)
        duration = onset.where(mask, 0.0).amax(-1, keepdim=True).expand_as(onset)
        rate = (duration / last).clamp(min=SHORTEST).div(HOLDOUT_GAP).log()
        span = [place.log1p(), length.log1p(), place / last, (length - 1 - place).clamp(min=0).log1p(), rate]
        alone = length.eq(1).to(onset.dtype)
        return torch.cat([*gaps, torch.stack([*span, alone], dim=-1)], dim=-1) * mask[..., None]

    @override
    def forward(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "..."]:
        """Each position's timing logits; a scalar 0 when inactive."""
        return self.layers(self.features(onset, mask)) if self.active else onset.new_zeros(())

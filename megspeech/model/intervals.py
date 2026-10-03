"""What a word is told of the gaps between its span's onsets, which bound its duration."""

from enum import StrEnum

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

HOLDOUT_GAP = 0.21
"""Median seconds between consecutive vocabulary words in a holdout sentence."""

SHORTEST = 0.02
"""One 50 Hz sample: two onsets never fall closer."""


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

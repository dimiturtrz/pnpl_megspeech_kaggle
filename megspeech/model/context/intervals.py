"""What timing beyond the clock each word of a span is told, every piece of it in the Kaggle test's inputs."""

from enum import StrEnum

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.context.timing import FEATURES, KAGGLE_GAP, SHORTEST, TimingHead


class Intervals(StrEnum):
    """What timing beyond the clock each word is told, every piece of it in the Kaggle test's inputs.

    `log`: the log gaps to the next and previous word's onsets, in Kaggle test gaps, each beside whether that neighbour
    exists. A word's duration identifies it far above chance (arXiv 2609.40359: log inter-onset time alone 22.9%
    top-1 of 50); a span's last word has no next onset, in training and Kaggle test alike.
    `rich`: the timing head's onset features (`TimingHead.features`): gaps to two neighbours each side, place in the
    span, its length and the span's onset rate.
    `ended`: `rich` and the span's end (`Reader.ending`): the time until its last word ends, the last word's own
    duration, and the words per second over the whole span. The Kaggle test reads the end from a sentence epoch's
    length (`sentence_n_times` less `sentence_extra_end_s` past the last onset); an isolated word's is unknown."""

    NONE = "none"
    LOG = "log"
    RICH = "rich"
    ENDED = "ended"

    @property
    def width(self) -> int:
        return WIDTHS[self]

    def features(
        self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], finish: Float[Tensor, " g"] | None = None
    ) -> Float[Tensor, "g w k"]:
        """Each word's features; `finish` is each span's last word's end in seconds from its first onset, unknown
        (all of them) when None and wherever it does not pass the last onset."""
        ends = onset.new_zeros(len(onset)) if finish is None else finish
        return READERS[self](onset, mask, ends)


ENDING = 3
"""`Reader.ending` per word: the log time to the span's end, whether it is known, and the span's log pace."""


class Reader:
    """Each `Intervals` member's features, one static method apiece, and what a span's end tells its words."""

    @staticmethod
    def none(
        onset: Float[Tensor, "g w"], _mask: Bool[Tensor, "g w"], _finish: Float[Tensor, " g"]
    ) -> Float[Tensor, "g w k"]:
        return onset.new_zeros(*onset.shape, 0)

    @staticmethod
    def log(
        onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], _finish: Float[Tensor, " g"]
    ) -> Float[Tensor, "g w k"]:
        """The log gaps to the next and previous onsets, each beside whether that neighbour exists."""
        pair = mask[:, 1:] & mask[:, :-1]
        gap = torch.where(pair, (onset[:, 1:] - onset[:, :-1]).clamp(min=SHORTEST) / KAGGLE_GAP, 1.0).log()
        pad = nn.functional.pad
        known = pair.to(onset.dtype)
        return torch.stack([pad(gap, (0, 1)), pad(known, (0, 1)), pad(gap, (1, 0)), pad(known, (1, 0))], dim=-1)

    @staticmethod
    def rich(
        onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], _finish: Float[Tensor, " g"]
    ) -> Float[Tensor, "g w k"]:
        return TimingHead.features(onset, mask)

    @staticmethod
    def ended(
        onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], finish: Float[Tensor, " g"]
    ) -> Float[Tensor, "g w k"]:
        return torch.cat([TimingHead.features(onset, mask), Reader.ending(onset, mask, finish)], dim=-1)

    @staticmethod
    def ending(
        onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"], finish: Float[Tensor, " g"]
    ) -> Float[Tensor, "g w k"]:
        """Each word's log time from its onset to the span's end in Kaggle test gaps (the last word's: its duration),
        whether the end is known, and the log words per Kaggle test gap over the whole span; zeros where unknown."""
        length = mask.sum(-1)
        last = onset.gather(1, (length - 1).clamp(min=0)[:, None])[:, 0]
        known = (finish > last)[:, None] & mask
        left = ((finish[:, None] - onset).clamp(min=SHORTEST) / KAGGLE_GAP).log()
        pace = (length.to(onset.dtype) * KAGGLE_GAP / finish.clamp(min=SHORTEST)).log()[:, None].expand_as(onset)
        flag = known.to(onset.dtype)
        return torch.stack([left.where(known, 0.0), flag, pace.where(known, 0.0)], dim=-1)


WIDTHS = {Intervals.NONE: 0, Intervals.LOG: 4, Intervals.RICH: FEATURES, Intervals.ENDED: FEATURES + ENDING}
READERS = {
    Intervals.NONE: Reader.none,
    Intervals.LOG: Reader.log,
    Intervals.RICH: Reader.rich,
    Intervals.ENDED: Reader.ended,
}

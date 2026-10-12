"""What timing beyond the clock each word of a span is told, every piece of it in the Kaggle test's inputs."""

from collections.abc import Callable
from dataclasses import dataclass
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
        return READERS[self](Reader(onset, mask, ends))


@dataclass(frozen=True)
class NextDrop:
    """Probability `p` that a word with a next neighbour reads its features as the last word of its span cut short
    after it, as training noise toward the Kaggle test, where a quarter of words (an epoch's last, an isolated word)
    have none. The cut span's end is unknown. 0 leaves every word's view whole."""

    p: float = 0.0

    def apply(
        self,
        intervals: Intervals,
        onset: Float[Tensor, "g w"],
        mask: Bool[Tensor, "g w"],
        finish: Float[Tensor, " g"] | None = None,
    ) -> Float[Tensor, "g w k"]:
        """`intervals`' features of each word, a drawn `p` of the words that have a next one read as cut after it."""
        whole = intervals.features(onset, mask, finish)
        if self.p == 0.0:
            return whole
        place = torch.arange(onset.shape[1], device=onset.device)
        followed = mask & (place < mask.sum(-1, keepdim=True) - 1)
        row, word = (followed & (torch.rand_like(onset) < self.p)).nonzero(as_tuple=True)
        cut = intervals.features(onset[row], mask[row] & (place <= word[:, None]))
        return whole.index_put((row, word), cut[torch.arange(len(word), device=onset.device), word])


ENDING = 3
"""`Reader.ending` per word: the log time to the span's end, whether it is known, and the span's log pace."""


@dataclass(frozen=True)
class Reader:
    """A span's words' onsets, which of them are real and its last word's end: each `Intervals` member's features,
    one method apiece, and what that end tells its words."""

    onset: Float[Tensor, "g w"]
    mask: Bool[Tensor, "g w"]
    finish: Float[Tensor, " g"]

    def none(self) -> Float[Tensor, "g w k"]:
        return self.onset.new_zeros(*self.onset.shape, 0)

    def log(self) -> Float[Tensor, "g w k"]:
        """The log gaps to the next and previous onsets, each beside whether that neighbour exists."""
        onset, mask = self.onset, self.mask
        pair = mask[:, 1:] & mask[:, :-1]
        gap = torch.where(pair, (onset[:, 1:] - onset[:, :-1]).clamp(min=SHORTEST) / KAGGLE_GAP, 1.0).log()
        pad = nn.functional.pad
        known = pair.to(onset.dtype)
        return torch.stack([pad(gap, (0, 1)), pad(known, (0, 1)), pad(gap, (1, 0)), pad(known, (1, 0))], dim=-1)

    def rich(self) -> Float[Tensor, "g w k"]:
        return TimingHead.features(self.onset, self.mask)

    def ended(self) -> Float[Tensor, "g w k"]:
        return torch.cat([self.rich(), self.ending()], dim=-1)

    def ending(self) -> Float[Tensor, "g w k"]:
        """Each word's log time from its onset to the span's end in Kaggle test gaps (the last word's: its duration),
        whether the end is known, and the log words per Kaggle test gap over the whole span; zeros where unknown."""
        onset, mask, finish = self.onset, self.mask, self.finish
        length = mask.sum(-1)
        last = onset.gather(1, (length - 1).clamp(min=0)[:, None])[:, 0]
        known = (finish > last)[:, None] & mask
        left = ((finish[:, None] - onset).clamp(min=SHORTEST) / KAGGLE_GAP).log()
        pace = (length.to(onset.dtype) * KAGGLE_GAP / finish.clamp(min=SHORTEST)).log()[:, None].expand_as(onset)
        flag = known.to(onset.dtype)
        return torch.stack([left.where(known, 0.0), flag, pace.where(known, 0.0)], dim=-1)


WIDTHS = {Intervals.NONE: 0, Intervals.LOG: 4, Intervals.RICH: FEATURES, Intervals.ENDED: FEATURES + ENDING}
READERS: dict[Intervals, Callable[[Reader], Float[Tensor, "g w k"]]] = {
    Intervals.NONE: Reader.none,
    Intervals.LOG: Reader.log,
    Intervals.RICH: Reader.rich,
    Intervals.ENDED: Reader.ended,
}

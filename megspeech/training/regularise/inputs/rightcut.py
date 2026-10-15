"""Right-context dropout: a training span now and then cut short after a random word, so that word ends its span.

A Kaggle test sentence epoch is a piece of a sentence, so its last row has no following word to read; the context
head, trained on whole sentences, scores such rows far below mid-span ones. Cutting a random share of training spans
at a random word makes every word sometimes last and usually not, where training on Kaggle-like cuts fixes each
word's place once.
"""

import logging
from dataclasses import dataclass, field, replace
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from megspeech.data.batching.split import Batch, Split
from megspeech.data.constants import RATE
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD

LOG = logging.getLogger(__name__)
SHORTEST = 3
"""Words a span needs to be cut: the cut keeps at least two, so no span turns lone, and drops at least one."""
TALLIES = ("cut", "spans", "last", "labelled")
"""What `RightCut` counts per step: spans cut, spans drawn, labelled words ending their span, labelled words."""


class Whole:
    """No cut: every span keeps all its words, no random draw made."""

    def apply(self, batch: Batch, train: Split) -> Batch:
        return batch

    def drained(self) -> dict[str, float]:
        """What was cut since the last call: nothing to report."""
        return {}


@dataclass
class RightCut(Whole):
    """Each span of at least `SHORTEST` words, with probability `p`, keeps only its words up to one drawn uniformly
    from its second to its second last; the words past it leave the batch for this step, and the span's `finish`
    becomes the kept last word's next onset (`Split.ended` with no tail)."""

    p: float
    rate: float = RATE
    tally: Float[Tensor, " 4"] = field(default_factory=lambda: torch.zeros(len(TALLIES)))

    @classmethod
    def of(cls, train: Split, share: float, rate: float = RATE) -> Whole:
        """Cutting at the `p` that makes `share` of `train`'s labelled words, as spans are drawn, end their span;
        none when `share` is 0."""
        if not share:
            return Whole()
        weight = train.weight[train.index[:, 0]]
        p, natural = cls.chance(train.index, train.y != IGNORE, weight, share)
        LOG.info("right cut: %.4f of spans for a last-word share %.4f (uncut %.4f)", p, share, natural)
        return cls(p, rate, torch.zeros(len(TALLIES), device=train.y.device))

    @staticmethod
    def chance(
        index: Int[Tensor, "g w"], labelled: Bool[Tensor, " n"], weight: Float[Tensor, " g"], share: float
    ) -> tuple[float, float]:
        """The cut probability whose expected share of labelled words ending their span is `share`, in [0, 1], and
        the uncut share. Spans are drawn at `weight`; both the labelled words and those ending a span are linear in
        the probability, so the share is a ratio of two lines, solved exactly."""
        valid = index != PAD
        marked = (valid & labelled[index.clamp(min=0)]).double()
        length = valid.sum(dim=1)
        ends = torch.arange(index.shape[1], device=index.device)
        middle = ((ends >= 1) & (ends[None] <= (length - 2)[:, None])).double()
        draws = middle.sum(dim=1).clamp(min=1)
        whole = torch.stack([marked.sum(dim=1), marked.gather(1, (length - 1).clamp(min=0)[:, None])[:, 0]])
        cut = torch.stack([(marked.cumsum(dim=1) * middle).sum(dim=1), (marked * middle).sum(dim=1)]) / draws
        weight = weight.double()
        base = (whole * weight).sum(dim=1)
        lift = ((cut - whole) * weight * (length >= SHORTEST)).sum(dim=1)
        rise = lift[1] - share * lift[0]
        p = torch.where(rise > 0, (share * base[0] - base[1]) / rise.clamp(min=1e-12), 0.0).clamp(0.0, 1.0)
        return float(p), float(base[1] / base[0].clamp(min=1e-12))

    @override
    def apply(self, batch: Batch, train: Split) -> Batch:
        valid = batch.mask
        length = valid.sum(dim=1)
        spans, width = valid.shape
        end = 1 + (torch.rand(spans, device=valid.device) * (length - 2).clamp(min=0)).long()
        cut = (length >= SHORTEST) & (torch.rand(spans, device=valid.device) < self.p)
        kept = valid & ~(cut[:, None] & (torch.arange(width, device=valid.device) > end[:, None]))
        index = torch.where(kept, batch.index, PAD)
        short = replace(
            batch,
            slot=torch.where(kept, batch.slot, len(batch.x)),
            index=index,
            finish=self.finished(batch, index, cut, train),
        )
        self.count(short, cut, train)
        return short

    def finished(
        self, batch: Batch, index: Int[Tensor, "g w"], cut: Bool[Tensor, " g"], train: Split
    ) -> Float[Tensor, " g"] | None:
        """Each span's `finish`: a cut span's from its kept last word's next onset, unknown where `batch`'s is."""
        if batch.finish is None:
            return None
        short = Split.ended(train.windows, index, batch.onset, self.rate, tail=0.0)
        return torch.where(cut, short, batch.finish)

    def count(self, batch: Batch, cut: Bool[Tensor, " g"], train: Split) -> None:
        """Adds `batch`'s spans cut and drawn, and its labelled words ending their span and in all, to the tally."""
        valid = batch.mask
        labelled = batch.target(train.y) != IGNORE
        last = torch.arange(valid.shape[1], device=valid.device) == (valid.sum(dim=1) - 1)[:, None]
        counts = torch.stack([cut.sum(), valid.any(dim=1).sum(), (labelled & last).sum(), labelled.sum()])
        self.tally.add_(counts.to(self.tally.dtype))

    @override
    def drained(self) -> dict[str, float]:
        """Spans cut and drawn, labelled words ending their span and in all, and that share, since the last call."""
        counts = dict(zip(TALLIES, self.tally.tolist(), strict=True))
        self.tally.zero_()
        return counts | {"share": counts["last"] / max(counts["labelled"], 1.0)}

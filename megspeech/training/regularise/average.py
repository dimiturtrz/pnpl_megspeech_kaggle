"""Same-word trial averaging in training (MEGConformer's dynamic grouping): a word's window becomes the mean of its own
and other windows of the same word, evoked SNR up, the neighbouring words' responses averaged out."""

from dataclasses import dataclass, field
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.split import Batch, Split
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.training.regularise.bgmix import BGMix, Untouched

UNUSED = 2.0
"""A sort key above every uniform draw: positions no partner may take."""


@dataclass
class GroupAverage(Untouched):
    """Each labelled word of a training batch, with probability `p`, becomes the mean of its own window and up to
    `k - 1` windows of other words with the same (subject, label), drawn afresh each step from the training split.

    A word with fewer same-subject repeats averages the ones that exist, a lone one stays itself. The label, onset and
    span stay; only the window's content changes, and inference reads none of this.
    """

    k: int
    p: float
    order: Int[Tensor, " m"]
    """The labelled words, grouped by (subject, label)."""
    start: Int[Tensor, " n"]
    """Where each word's group begins in `order`."""
    size: Int[Tensor, " n"]
    """Each word's group size; 0 for a word outside the vocabulary."""
    rank: Int[Tensor, " n"]
    """Each word's own position in `order`."""
    mixed: Tensor = field(default_factory=lambda: torch.zeros((), dtype=torch.long))

    @classmethod
    def of(cls, train: Split, k: int, p: float) -> Untouched:
        """Averaging `k` windows at probability `p` over `train`'s labelled words; none when `k` is 1 or `p` is 0."""
        if k <= 1 or p <= 0.0:
            return Untouched()
        words = train.labelled()
        key = train.subject[words] * VOCAB_SIZE + train.y[words]
        grouped = key.argsort(stable=True)
        order, key = words[grouped], key[grouped]
        cells = torch.bincount(key)
        first = cells.cumsum(0) - cells
        start, size, rank = (torch.zeros_like(train.y) for _ in range(3))
        start[order], size[order], rank[order] = first[key], cells[key], torch.arange(len(order), device=key.device)
        zero = torch.zeros((), dtype=torch.long, device=key.device)
        return cls(k, p, order, start, size, rank, zero)

    @override
    def apply(self, x: Float[Tensor, "r c t"], batch: Batch, train: Split) -> Float[Tensor, "r c t"]:
        """`x` with each chosen labelled row's window averaged with its same-word partners'."""
        rows = BGMix.words(batch)
        word = rows.clamp(min=0)
        size, start = self.size[word], self.start[word]
        positions = torch.arange(int(self.size.max()), device=x.device)
        draw = torch.rand(len(x), len(positions), device=x.device)
        taken = (positions >= size[:, None]) | (positions == (self.rank[word] - start)[:, None])
        values, offset = torch.where(taken, UNUSED, draw).topk(min(self.k - 1, len(positions)), dim=1, largest=False)
        usable = (values < UNUSED) & (rows >= 0)[:, None]
        partners = self.order[(start[:, None] + offset).clamp(max=len(self.order) - 1)]
        windows = train.windows(partners.flatten()).float().view(*partners.shape, *x.shape[1:])
        total = x.float() + (windows * usable[:, :, None, None]).sum(dim=1)
        mean = total / (1 + usable.sum(dim=1))[:, None, None]
        chosen = usable.any(dim=1) & (torch.rand(len(x), device=x.device) < self.p)
        self.mixed += chosen.sum()
        return torch.where(chosen[:, None, None], mean.to(x.dtype), x)

    @override
    def drained(self) -> int:
        count = int(self.mixed.item())
        self.mixed.zero_()
        return count

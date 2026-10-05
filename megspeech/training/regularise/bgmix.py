"""Trial-level BGMix (PMC12501431): a labelled word's window swaps its own background for another word's."""

from dataclasses import dataclass, field
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.split import Batch, Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE

MIN_CELL = 20
"""Words a (subject, label) cell needs for its own template; fewer take the label's template over every subject."""
CHUNK = 2048
"""Windows read at a time when averaging the templates."""


@dataclass
class Untouched:
    """No mixing: windows pass through, no random draw made."""

    def apply(self, x: Float[Tensor, "r c t"], batch: Batch, train: Split) -> Float[Tensor, "r c t"]:
        return x

    def drained(self) -> int:
        """Words mixed since the last call, always none."""
        return 0


@dataclass
class BGMix(Untouched):
    """Each labelled word of a training batch, with probability `p`, becomes its own (subject, label) template plus the
    residual of another labelled word of the batch: the other's window less the other's template. The label stays.

    Templates come from training words alone and live outside the model, so inference reads none of this.
    """

    p: float
    table: Float[Tensor, "s v c t"]
    mixed: Tensor = field(default_factory=lambda: torch.zeros((), dtype=torch.long))

    @classmethod
    def of(cls, train: Split, p: float) -> Untouched:
        """Mixing at `p` on templates averaged over `train`'s labelled windows; none when `p` is 0."""
        if not p:
            return Untouched()
        words = train.labelled()
        subjects = int(train.subject.max()) + 1
        sums = torch.zeros(0, device=train.y.device)
        cells = torch.zeros(subjects * VOCAB_SIZE, device=train.y.device)
        for chunk in words.split(CHUNK):
            windows = train.windows(chunk).float()
            cell = train.subject[chunk] * VOCAB_SIZE + train.y[chunk]
            if not sums.numel():
                sums = torch.zeros(subjects * VOCAB_SIZE, *windows.shape[1:], device=windows.device)
            sums.index_add_(0, cell, windows)
            cells += torch.bincount(cell, minlength=len(cells))
        table = cls.averaged(sums, cells, subjects)
        return cls(p, table, torch.zeros((), dtype=torch.long, device=table.device))

    @staticmethod
    def averaged(sums: Float[Tensor, "k c t"], counts: Float[Tensor, " k"], subjects: int) -> Float[Tensor, "s v c t"]:
        """Each (subject, label) cell's mean window, the label's mean over every subject where it holds under
        `MIN_CELL` words."""
        own = sums / counts.clamp(min=1)[:, None, None]
        per_label = sums.view(subjects, VOCAB_SIZE, *sums.shape[1:]).sum(0)
        pooled = per_label / counts.view(subjects, VOCAB_SIZE).sum(0).clamp(min=1)[:, None, None]
        thin = (counts < MIN_CELL).view(subjects, VOCAB_SIZE, 1, 1)
        return torch.where(thin, pooled[None], own.view(subjects, VOCAB_SIZE, *sums.shape[1:]))

    @staticmethod
    def rows(batch: Batch, train: Split) -> Int[Tensor, " r"]:
        """Each row's label, `IGNORE` for padding and for words outside the vocabulary."""
        rows = len(batch.x)
        word = batch.index.new_full((rows + 1,), -1).scatter_(0, batch.slot.flatten(), batch.index.flatten())[:rows]
        return torch.where(word >= 0, train.y[word.clamp(min=0)], IGNORE)

    @override
    def apply(self, x: Float[Tensor, "r c t"], batch: Batch, train: Split) -> Float[Tensor, "r c t"]:
        """`x` with each chosen labelled row's window replaced as the class says."""
        label = self.rows(batch, train)
        known = label != IGNORE
        eye = torch.eye(len(x), dtype=torch.bool, device=x.device)
        others = known[None, :] & ~eye
        others |= eye & ~others.any(dim=1, keepdim=True)
        partner = torch.multinomial(others.float(), 1).squeeze(1)
        chosen = (
            known & (torch.rand(len(x), device=x.device) < self.p) & (partner != torch.arange(len(x), device=x.device))
        )
        label = label.clamp(min=0)
        own = self.table[batch.subject, label]
        theirs = self.table[batch.subject[partner], label[partner]]
        swapped = own + x[partner].to(own.dtype) - theirs
        self.mixed += chosen.sum()
        return torch.where(chosen[:, None, None], swapped.to(x.dtype), x)

    @override
    def drained(self) -> int:
        count = int(self.mixed.item())
        self.mixed.zero_()
        return count

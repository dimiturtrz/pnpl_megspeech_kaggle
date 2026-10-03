"""A fixed-shape training batch: spans packed into a set number of word rows."""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.labels.vocabulary import IGNORE


@dataclass(frozen=True)
class Batch:
    """Spans packed into a fixed number of word rows, so every training step has the same shapes.

    `slot` is each span position's row of `x`, `len(x)` where the position is empty; `index` its split word.
    """

    x: Float[Tensor, "r c t"]
    onset: Float[Tensor, "g w"]
    slot: Int[Tensor, "g w"]
    index: Int[Tensor, "g w"]
    subject: Int[Tensor, " r"]

    def target(self, labels: Int[Tensor, " n"]) -> Int[Tensor, "g w"]:
        """Each position's label from the split's `labels`; `IGNORE` where the position is empty."""
        return torch.where(self.slot < len(self.x), labels[self.index.clamp(min=0)], IGNORE)

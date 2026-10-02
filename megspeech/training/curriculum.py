"""How training draws spans across subjects as it goes: their natural mix, or annealed toward the Deep subject."""

from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.data.runs import DEEP
from megspeech.data.split import Split


class Curriculum(Protocol):
    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        """Each span's draw weight in `epoch` of `epochs`."""
        ...


class Natural:
    """Every span at its own weight, whichever subject heard it: the mix the corpus holds."""

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        return train.weight[train.index[:, 0]]


@dataclass(frozen=True)
class Annealed:
    """Subjects drawn equally at the first epoch, the Deep subject's share then rising linearly to `share` at the
    last; the others split the rest equally, and within a subject each span keeps its own weight.

    Many subjects early regularise the shared encoder toward what every head has in common; the Deep subject late
    fits it to the head it is scored on.
    """

    share: float

    @classmethod
    def of(cls, share: float) -> Curriculum:
        """Annealed toward `share`; the natural mix at 0."""
        return cls(share) if share else Natural()

    def weights(self, train: Split, epoch: int, epochs: int) -> Float[Tensor, " g"]:
        first = train.index[:, 0]
        weight, subject = train.weight[first], train.subject[first]
        present, which = torch.unique(subject, return_inverse=True)
        totals = torch.zeros(len(present), dtype=weight.dtype, device=weight.device).index_add_(0, which, weight)
        progress = epoch / max(epochs - 1, 1)
        deep = (1 - progress) / len(present) + progress * self.share
        others = (1 - deep) / max(len(present) - 1, 1)
        shares = torch.where(present == DEEP, deep, others)
        return weight / totals[which] * shares[which]

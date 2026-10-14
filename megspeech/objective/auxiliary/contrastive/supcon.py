"""The supervised-contrastive core (Khosla et al. 2020, `L_out`) the auxiliary terms that pull embeddings
together share: an anchor's loss is the log-sum-exp over its candidates less the mean logit of its positives,
averaged over the anchors with any. What a candidate and a positive are is the term's `Positives` strategy; the rest
is here.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.auxiliary.contrastive.similarity import Similarity
from megspeech.objective.term import AuxiliaryTerm, Reading


@dataclass(frozen=True)
class Positives(ABC):
    """Which pairs of a batch's word slots are candidates and which of those positives."""

    @abstractmethod
    def pairs(self, target: Int[Tensor, " m"]) -> tuple[Bool[Tensor, "m m"], Bool[Tensor, "m m"]]:
        """Over the flattened slots carrying `target` (`IGNORE` where a slot holds no labelled word), the pairs that
        are candidates and, among those, the positives."""


@dataclass(frozen=True)
class SupConTerm(AuxiliaryTerm, ABC):
    """Supervised contrastive loss over the batch's word embeddings, its candidates and positives `positives`."""

    @abstractmethod
    def positives(self) -> Positives:
        """The strategy naming this term's candidates and positives."""

    def pairs(self, target: Int[Tensor, " m"]) -> tuple[Bool[Tensor, "m m"], Bool[Tensor, "m m"]]:
        """`positives`' candidate and positive pairs over the slots carrying `target`."""
        return self.positives().pairs(target)

    @override
    def head(self, width: int) -> nn.Module:
        return Similarity()

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        flat = reading.target.flatten()
        candidate, positive = self.pairs(flat)
        output = reading.output
        with torch.autocast(output.device.type, enabled=False):
            unit = functional.normalize(output.flatten(0, 1).float(), dim=-1)
            logits = unit @ unit.T * cast("Similarity", reading.head).scale((flat != IGNORE).sum())
        open_ = candidate.any(dim=1, keepdim=True)
        spread = torch.where(open_, logits.masked_fill(~candidate, -math.inf), 0.0).logsumexp(dim=1)
        count = positive.sum(dim=1)
        pulled = (logits * positive).sum(dim=1) / count.clamp(min=1)
        return self.averaged(spread - pulled, count > 0, reading.rows.flatten())

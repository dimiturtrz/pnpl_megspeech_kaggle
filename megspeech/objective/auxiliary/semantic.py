"""The semantic auxiliary: each heard word's embedding aligned to its t5 embedding by in-batch SigLIP."""

from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.objective.scorer import SigmoidScorer
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import AuxiliaryTerm, Reading


class Semantic(SigmoidScorer):
    """A projection of word embeddings into t5-large's space, scored against the batch's distinct words."""

    @override
    def forward(self, embedded: Float[Tensor, "*b w"]) -> Float[Tensor, "*b d"]:
        return self.project(embedded)

    def loss(
        self,
        projected: Float[Tensor, "n d"],
        targets: Float[Tensor, "n d"],
        kinds: Int[Tensor, " n"],
        rows: Float[Tensor, " n"],
    ) -> Float[Tensor, ""]:
        """SigLIP of each row against every distinct kind in the batch, positive on its own kind only.

        A negative kind is no word (padding, or a word without a target): it is neither a row nor a candidate.
        Candidates are deduplicated by kind without a host sync, so the step keeps fixed shapes. The rows
        average at their weights in `rows`.
        """
        valid = kinds >= 0
        same = kinds[:, None] == kinds[None, :]
        candidate = valid & ~same.tril(-1).any(dim=1)
        logits = self.score(projected, functional.normalize(targets.float(), dim=-1))
        pairs = functional.binary_cross_entropy_with_logits(logits, same.float(), reduction="none")
        return AuxiliaryTerm.averaged((pairs * candidate).sum(-1), valid, rows)

    def softmax_loss(
        self,
        projected: Float[Tensor, "n d"],
        targets: Float[Tensor, "n d"],
        kinds: Int[Tensor, " n"],
        rows: Float[Tensor, " n"],
    ) -> Float[Tensor, ""]:
        """Cross-entropy of each row over the batch's distinct kinds, its own the label: the contrast learns like a
        classifier over the batch's kinds (as the 50-way heads do), where `loss`'s independent sigmoids plateau at
        the constant answer. Kinds are deduplicated as in `loss`."""
        valid = kinds >= 0
        same = kinds[:, None] == kinds[None, :]
        candidate = valid & ~same.tril(-1).any(dim=1)
        logits = self.score(projected, functional.normalize(targets.float(), dim=-1))
        logits = logits.masked_fill(~candidate[None, :], torch.finfo(logits.dtype).min)
        own = (logits.log_softmax(-1) * (same & candidate[None, :])).sum(-1)
        return -AuxiliaryTerm.averaged(own, valid, rows)


@dataclass(frozen=True)
class SemanticTerm(AuxiliaryTerm):
    """Every heard word's window, in or out of the 50, trained toward its t5-large embedding (`Semantic`)."""

    weight: float
    lexicon: Lexicon

    @override
    def head(self, width: int) -> nn.Module:
        return Semantic(width).to(self.lexicon.vectors.device)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        semantic = cast("Semantic", reading.head)
        return reading.contrasted(semantic.loss, self.lexicon.vectors, self.lexicon.rows(reading.target))

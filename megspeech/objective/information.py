"""The OVMI auxiliary: the context head's loss with minus the batch's soft mutual information between word and
prediction added, the differentiable twin of `Ruler.information`."""

from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.term import ContextTerm, Term

EPSILON = 1e-12
"""Floor under every probability before its log, so an absent cell contributes 0 and no NaN gradient."""


@dataclass(frozen=True)
class InformedTerm(Term):
    """A `base` context term plus `weight` times minus the mutual information in bits of the batch's labelled words
    (`information`), the confusion rows weighted by `prior` renormalised over the classes present."""

    base: ContextTerm
    prior: Float[Tensor, " v"]

    @staticmethod
    def of(base: ContextTerm, weight: float, prior: Float[Tensor, " v"]) -> ContextTerm | Term:
        """`base` itself at weight 0, so nothing is created."""
        return InformedTerm(base=base, prior=prior, weight=weight) if weight else base

    @staticmethod
    def information(
        logits: Float[Tensor, "n v"], y: Int[Tensor, " n"], prior: Float[Tensor, " v"]
    ) -> Float[Tensor, ""]:
        """Mutual information in bits between the label and the softmax of `logits` over the words with a label:
        each present class's summed softmax normalised to a distribution, weighted by `prior` over the present."""
        known = y != IGNORE
        probabilities = logits.float().softmax(-1) * known[:, None]
        joint = torch.zeros(VOCAB_SIZE, VOCAB_SIZE, device=logits.device).index_add(0, y.clamp(min=0), probabilities)
        rows = torch.zeros(VOCAB_SIZE, device=logits.device).index_add(0, y.clamp(min=0), known.float())
        weight = torch.where(rows > 0, prior, 0.0)
        joint = joint / rows.clamp(min=1)[:, None] * (weight / weight.sum().clamp(min=EPSILON))[:, None]
        expected = joint.sum(dim=1, keepdim=True) * joint.sum(dim=0, keepdim=True)
        return (joint * (joint.clamp(min=EPSILON).log2() - expected.clamp(min=EPSILON).log2())).sum()

    @override
    def loss(
        self,
        head: nn.Module,
        output: Float[Tensor, "g w _"],
        target: Int[Tensor, "g w"],
        inside: Bool[Tensor, "g w"] | None = None,
    ) -> Float[Tensor, ""]:
        return self.base.loss(head, output, target, inside) + super().loss(head, output, target, inside)

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        return -self.information(output.flatten(0, 1), self.base.hard(target).flatten(), self.prior)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        return self.base.labels(split)

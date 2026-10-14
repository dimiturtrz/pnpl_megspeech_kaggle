"""The head of the supervised-contrastive terms
(`megspeech.objective.auxiliary.contrastive.supcon`): the embeddings unchanged and
the learned temperature of their cosine logits.

The logits are cosines times `exp(theta) * ln N` over the batch's `N` labelled words, `theta` learned from 0: at
`ln N` a positive at cosine 1 among `N - 1` negatives at cosine 0 holds half the softmax, where its gradient
`p (1 - p)` is largest, so the scale starts where the term learns fastest whatever the batch's size.
"""

from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class Similarity(nn.Module):
    """The embeddings unchanged, the term read off them directly; holds the learned `theta` of the logit scale."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(()))

    @override
    def forward(self, x: Float[Tensor, "g w d"]) -> Float[Tensor, "g w d"]:
        return x

    def scale(self, candidates: Int[Tensor, ""]) -> Float[Tensor, ""]:
        """The logit scale over `candidates` words: `exp(theta) * ln candidates`, at least `ln 2`."""
        return self.theta.exp() * candidates.clamp(min=2).float().log()

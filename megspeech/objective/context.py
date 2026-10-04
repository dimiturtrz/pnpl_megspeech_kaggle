"""The context head and its loss: a free 50-way softmax, or SigLIP retrieval against fixed word prototypes."""

from enum import StrEnum
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.scorer import DIM, SigmoidScorer
from megspeech.objective.topk import SmoothTopK


class CosineHead(SigmoidScorer):
    """Logits as scaled cosine similarity between a projected embedding and each word's frozen prototype."""

    prototypes: Tensor

    def __init__(self, width: int, prototypes: Float[Tensor, "v d"]) -> None:
        super().__init__(width, prototypes.shape[1])
        self.register_buffer("prototypes", functional.normalize(prototypes.float(), dim=-1))

    @override
    def forward(self, h: Float[Tensor, "*b w"]) -> Float[Tensor, "*b v"]:
        return self.score(self.project(h), self.prototypes)


class Objective(StrEnum):
    """How the context head scores words, and the loss that trains it."""

    SOFTMAX = "softmax"
    SIGLIP = "siglip"
    TOPK = "topk"

    def head(self, width: int, prototypes: Float[Tensor, "v d"] | None) -> nn.Module:
        """`prototypes` None stands in zeros, for a checkpoint that will overwrite them."""
        anchors = torch.zeros(VOCAB_SIZE, DIM) if prototypes is None else prototypes
        return CosineHead(width, anchors) if self is Objective.SIGLIP else nn.Linear(width, VOCAB_SIZE)

    def loss(
        self,
        logits: Float[Tensor, "n v"],
        y: Int[Tensor, " n"],
        smoothing: float,
        weight: Float[Tensor, " v"] | None = None,
    ) -> Float[Tensor, ""]:
        """Softmax cross-entropy, the smooth top-k loss over the same logits, or SigLIP's per-pair sigmoid loss summed
        over the 50 candidates.

        Rows labelled `IGNORE` are padding: they add nothing and do not count toward the mean. `weight` scales each
        class's rows in the cross-entropy and top-k loss; SigLIP scores every candidate per row and ignores it.
        """
        if self is Objective.TOPK:
            return SmoothTopK()(logits, y, weight)
        if self is Objective.SOFTMAX:
            return functional.cross_entropy(
                logits.float(), y, weight=weight, ignore_index=IGNORE, label_smoothing=smoothing
            )
        keep = (y != IGNORE).float()
        target = functional.one_hot(y.clamp(min=0), logits.shape[-1]).float()
        pairs = functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="none").sum(-1)
        return (pairs * keep).sum() / keep.sum()

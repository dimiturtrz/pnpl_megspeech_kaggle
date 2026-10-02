"""The context head and its loss: a free 50-way softmax, or SigLIP retrieval against fixed word prototypes."""

import math
from enum import StrEnum
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.wordnet import WORDS

DIM = 1024
"""Width of a t5-large hidden state, the prototypes' space."""
TEMPERATURE = math.log(10)
BIAS = -10.0
"""SigLIP's initial log-temperature and bias (Zhai et al. 2023), as d'Ascoli et al. use them."""
IGNORE = -100
"""The label of a padding row; `cross_entropy`'s default `ignore_index`."""


class CosineHead(nn.Module):
    """Logits as scaled cosine similarity between a projected embedding and each word's frozen prototype."""

    prototypes: Tensor

    def __init__(self, width: int, prototypes: Float[Tensor, "v d"]) -> None:
        super().__init__()
        self.project = nn.Linear(width, prototypes.shape[1])
        self.register_buffer("prototypes", functional.normalize(prototypes.float(), dim=-1))
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE))
        self.bias = nn.Parameter(torch.tensor(BIAS))

    @override
    def forward(self, h: Float[Tensor, "*b w"]) -> Float[Tensor, "*b v"]:
        similarity = functional.normalize(self.project(h).float(), dim=-1) @ self.prototypes.T
        return self.temperature.exp() * similarity + self.bias


class Objective(StrEnum):
    """How the context head scores words, and the loss that trains it."""

    SOFTMAX = "softmax"
    SIGLIP = "siglip"

    def head(self, width: int, prototypes: Float[Tensor, "v d"] | None) -> nn.Module:
        """`prototypes` None stands in zeros, for a checkpoint that will overwrite them."""
        anchors = torch.zeros(WORDS, DIM) if prototypes is None else prototypes
        return nn.Linear(width, WORDS) if self is Objective.SOFTMAX else CosineHead(width, anchors)

    def loss(self, logits: Float[Tensor, "n v"], y: Int[Tensor, " n"], smoothing: float) -> Float[Tensor, ""]:
        """Softmax cross-entropy, or SigLIP's per-pair sigmoid loss summed over the 50 candidates.

        Rows labelled `IGNORE` are padding: they add nothing and do not count toward the mean.
        """
        if self is Objective.SOFTMAX:
            return functional.cross_entropy(logits.float(), y, ignore_index=IGNORE, label_smoothing=smoothing)
        keep = (y != IGNORE).float()
        target = functional.one_hot(y.clamp(min=0), logits.shape[-1]).float()
        pairs = functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="none").sum(-1)
        return (pairs * keep).sum() / keep.sum()

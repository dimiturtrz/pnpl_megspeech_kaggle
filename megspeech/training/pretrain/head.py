"""The prototype head: features to logits over a bank of learned prototypes."""

import math
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from megspeech.training.pretrain.prototypes import Prototypes


class Head(nn.Module):
    """Features to prototype logits: layer-norm, an MLP to a bottleneck, then cosine similarity to K prototypes (a
    weight-normalised layer whose gain is fixed at 1, as DINO freezes it)."""

    def __init__(self, width: int, config: Prototypes) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.bottleneck),
        )
        self.prototypes = nn.Parameter(torch.randn(config.k, config.bottleneck) / math.sqrt(config.bottleneck))

    @override
    def forward(self, features: Float[Tensor, "n w"]) -> Float[Tensor, "n k"]:
        unit = functional.normalize(self.body(features).float(), dim=-1)
        return unit @ functional.normalize(self.prototypes, dim=-1).T

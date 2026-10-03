"""The sigmoid scorer SigLIP heads share: a learned projection, scored by scaled cosine against fixed anchors."""

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

DIM = 1024
"""Width of a t5-large hidden state, the prototypes' space."""
TEMPERATURE = math.log(10)
BIAS = -10.0
"""SigLIP's initial log-temperature and bias (Zhai et al. 2023), as d'Ascoli et al. use them."""


class SigmoidScorer(nn.Module):
    """A projection of `width`-wide embeddings into a `dim`-wide space, and the learned temperature and bias that
    turn its cosine similarity to unit-norm anchors into sigmoid logits."""

    def __init__(self, width: int, dim: int = DIM) -> None:
        super().__init__()
        self.project = nn.Linear(width, dim)
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE))
        self.bias = nn.Parameter(torch.tensor(BIAS))

    def score(self, projected: Float[Tensor, "*b d"], anchors: Float[Tensor, "v d"]) -> Float[Tensor, "*b v"]:
        """`projected` against each of the unit-norm `anchors`: `exp(temperature) * cosine + bias`."""
        similarity = functional.normalize(projected.float(), dim=-1) @ anchors.T
        return self.temperature.exp() * similarity + self.bias

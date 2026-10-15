"""Learned attention queries over a word window's time axis: an alternative word readout."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class AttentionPool(nn.Module):
    """One softmax over samples per learned query, each a weighted mean of the hidden channels."""

    def __init__(self, hidden: int, queries: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(hidden, queries, 1)

    @override
    def forward(self, h: Float[Tensor, "b d t"]) -> Float[Tensor, "b e"]:
        weights = self.score(h).softmax(dim=-1)
        return torch.einsum("bqt,bdt->bqd", weights, h).flatten(1)

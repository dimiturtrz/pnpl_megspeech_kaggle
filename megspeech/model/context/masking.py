"""Whole words hidden from the span transformer in training (BERT's masking at the word level)."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

MASK_INIT = 0.02
"""Spread of the mask embedding at initialisation, a transformer's usual embedding scale (BERT, GPT-2)."""


class WordMask(nn.Module):
    """Each valid word's token replaced with probability `rate` by one learned embedding, in training only; the
    caller adds the word's position after, so a masked word keeps its place. A rate of 0 has no parameter and
    passes the token through."""

    token: Tensor

    def __init__(self, width: int, rate: float) -> None:
        super().__init__()
        self.rate = rate
        if rate:
            self.token = nn.Parameter(torch.randn(width) * MASK_INIT)
        else:
            self.register_buffer("token", torch.zeros(width), persistent=False)

    @override
    def forward(self, content: Float[Tensor, "g w d"], valid: Bool[Tensor, "g w"]) -> Float[Tensor, "g w d"]:
        if not self.training or not self.rate:
            return content
        hit = (torch.rand(valid.shape, device=content.device) < self.rate) & valid
        return torch.where(hit[..., None], self.token, content)

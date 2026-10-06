"""Transformer layers across the site axis of a channel-first front end."""

from typing import override

from jaxtyping import Float
from torch import Tensor, nn


class SiteAttention(nn.Module):
    """Transformer layers over the site axis, each time step attending across sites independently."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    @override
    def forward(self, h: Float[Tensor, "b s d t"]) -> Float[Tensor, "b s d t"]:
        b, sites, dim, t = h.shape
        rows = h.permute(0, 3, 1, 2).reshape(b * t, sites, dim)
        return self.layers(rows).reshape(b, t, sites, dim).permute(0, 2, 3, 1)

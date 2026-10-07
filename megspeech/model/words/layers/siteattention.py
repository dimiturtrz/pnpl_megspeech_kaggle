"""Transformer layers across the site axis of a channel-first front end."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class SiteAttention(nn.Module):
    """Transformer layers over the site axis, each time step attending across sites independently; dropped sites
    (`keep` False) are masked out as keys. Attention runs on the math kernel, as in `SpanConformer`."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    @override
    def forward(self, h: Float[Tensor, "b s d t"], keep: Bool[Tensor, "b s"] | None = None) -> Float[Tensor, "b s d t"]:
        b, sites, dim, t = h.shape
        rows = h.permute(0, 3, 1, 2).reshape(b * t, sites, dim)
        mask = None if keep is None else ~keep.repeat_interleave(t, 0)
        with sdpa_kernel(SDPBackend.MATH):
            attended = self.layers(rows, src_key_padding_mask=mask)
        return attended.reshape(b, t, sites, dim).permute(0, 2, 3, 1)

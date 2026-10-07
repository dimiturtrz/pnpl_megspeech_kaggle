"""The layout-agnostic set-pool fuse of a channel-first front end."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class SetPool(nn.Module):
    """A layout-agnostic fuse: `queries` learned queries cross-attend over the site tokens at every time step
    (keys and values the tokens, dropped sites masked out), and the `queries * out` pooled channels go through a
    1x1 convolution to `hidden`. Attention runs on the math kernel, as in `SpanConformer`."""

    def __init__(self, out: int, queries: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, out) / out**0.5)
        self.attn = nn.MultiheadAttention(out, heads, batch_first=True)
        self.proj = nn.Conv1d(queries * out, hidden, 1)

    @override
    def forward(self, h: Float[Tensor, "b s d t"], keep: Bool[Tensor, "b s"] | None = None) -> Float[Tensor, "b h t"]:
        b, sites, dim, t = h.shape
        rows = h.permute(0, 3, 1, 2).reshape(b * t, sites, dim)
        mask = None if keep is None else ~keep.repeat_interleave(t, 0)
        queries = self.queries.expand(b * t, -1, -1)
        with sdpa_kernel(SDPBackend.MATH):
            pooled, _ = self.attn(queries, rows, rows, key_padding_mask=mask, need_weights=False)
        return self.proj(pooled.reshape(b, t, -1).transpose(1, 2))

"""Stochastic depth (Huang et al. 2016) on the span transformer's residual branches."""

from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """A residual branch zeroed for each sample with probability `rate` in training, the kept ones scaled by
    1/(1-rate) so the expectation holds; evaluation and a rate of 0 pass the branch through."""

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        self.rate = rate

    @override
    def forward(self, branch: Tensor) -> Tensor:
        if not self.training or not self.rate:
            return branch
        keep = torch.rand((len(branch),) + (1,) * (branch.dim() - 1), device=branch.device) >= self.rate
        return branch * keep.to(branch.dtype) / (1.0 - self.rate)

    def residual(self, base: Tensor, out: Tensor) -> Tensor:
        """`out`, a layer's output over `base`, with only its change over `base` droppable."""
        return base + self(out - base) if self.training and self.rate else out

    @staticmethod
    def ramp(paths: Iterable[nn.Module], peak: float) -> None:
        """Each of the `paths`' rates set linearly increasing, the last at `peak`."""
        ordered = [path for path in paths if isinstance(path, DropPath)]
        for depth, path in enumerate(ordered, start=1):
            path.rate = peak * depth / len(ordered)


class DropPathLayer(nn.TransformerEncoderLayer):
    """A pre-norm transformer layer whose attention and feed-forward branches each pass through `path` in training;
    with rate 0 or in evaluation exactly `nn.TransformerEncoderLayer`."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__(width, heads, feedforward, dropout, batch_first=True, norm_first=True)
        self.path = DropPath()

    @override
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if not self.training or not self.path.rate:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)
        x = src + self.path(self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal))
        return x + self.path(self._ff_block(self.norm2(x)))

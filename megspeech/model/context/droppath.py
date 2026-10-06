"""Stochastic depth (Huang et al. 2016) on the span transformer's residual branches."""

from collections.abc import Iterable
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class DropPath(nn.Module):
    """A residual branch zeroed for each sample with probability `rate` in training, the kept ones scaled by
    1/(1-rate) so the expectation holds; evaluation and a rate of 0 pass the branch through."""

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        self.rate = rate

    @override
    def forward(self, branch: Float[Tensor, "b *rest"]) -> Float[Tensor, "b *rest"]:
        if not self.training or not self.rate:
            return branch
        keep = torch.rand((len(branch),) + (1,) * (branch.dim() - 1), device=branch.device) >= self.rate
        return branch * keep.to(branch.dtype) / (1.0 - self.rate)

    def residual(self, base: Float[Tensor, "b *rest"], out: Float[Tensor, "b *rest"]) -> Float[Tensor, "b *rest"]:
        """`out`, a layer's output over `base`, with only its change over `base` droppable."""
        return base + self(out - base) if self.training and self.rate else out

    @staticmethod
    def ramp(paths: Iterable[nn.Module], peak: float) -> None:
        """Each of the `paths`' rates set linearly increasing, the last at `peak`."""
        ordered = [path for path in paths if isinstance(path, DropPath)]
        for depth, path in enumerate(ordered, start=1):
            path.rate = peak * depth / len(ordered)

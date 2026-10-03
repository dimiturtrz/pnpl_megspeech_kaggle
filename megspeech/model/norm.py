"""Batch normalisation whose statistics come from the valid rows only, so a fixed-size padded batch trains as the
unpadded one would."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional

EPS = 1e-5
MOMENTUM = 0.1
"""`nn.BatchNorm1d`'s defaults."""


class MaskedBatchNorm1d(nn.Module):
    """`nn.BatchNorm1d`'s parameters and buffers under its names, so either module loads the other's checkpoint."""

    running_mean: Tensor
    running_var: Tensor
    num_batches_tracked: Tensor

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("running_mean", torch.zeros(width))
        self.register_buffer("running_var", torch.ones(width))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """Batch statistics over the `valid` rows (all when None) in training, the running ones in evaluation."""
        if not self.training:
            return functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, eps=EPS)
        weight = (torch.ones(len(x), device=x.device) if valid is None else valid.float())[:, None, None]
        count = weight.sum() * x.shape[-1]
        h = x.float()
        mean = (h * weight).sum(dim=(0, 2)) / count
        var = ((h - mean[:, None]).square() * weight).sum(dim=(0, 2)) / count
        self.track(mean, var, count)
        normed = (h - mean[:, None]) * torch.rsqrt(var[:, None] + EPS)
        return (normed * self.weight[:, None] + self.bias[:, None]).to(x.dtype)

    def track(self, mean: Float[Tensor, " h"], var: Float[Tensor, " h"], count: Float[Tensor, ""]) -> None:
        """Running statistics as `nn.BatchNorm1d` keeps them: unbiased variance, exponential average. A batch of
        fewer than two valid samples has no unbiased variance (an all-padding one no mean), so it leaves them as
        they were, decided on the device without a host sync."""
        with torch.no_grad():
            kept = count > 1
            mean_step = self.running_mean.lerp(mean, MOMENTUM)
            var_step = self.running_var.lerp(var * count / (count - 1), MOMENTUM)
            self.running_mean.copy_(torch.where(kept, mean_step, self.running_mean))
            self.running_var.copy_(torch.where(kept, var_step, self.running_var))
            self.num_batches_tracked.add_(kept.long())

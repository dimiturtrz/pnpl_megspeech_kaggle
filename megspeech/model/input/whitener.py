"""Multivariate noise normalisation (Guggenmos, Sterzer & Cichy 2018): scaled MEG decorrelated across sensors by
the inverse square root of its shrunk covariance, fit on the training split."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Whitener(nn.Module):
    """ZCA whitening by a Ledoit-Wolf shrunk sensor covariance, each window a sample at each of its time points.

    A word's evoked field is a small share of a single trial's variance, so the windows' covariance stands in for
    the noise covariance Guggenmos et al. take from residuals; the rotation back to sensors keeps each channel where
    the subject layer and the encoder expect it.
    """

    matrix: Tensor

    def __init__(self, channels: int, *, active: bool = True) -> None:
        """An inactive whitener passes the sensors through and saves nothing, so models trained before it load."""
        super().__init__()
        self.active = active
        self.register_buffer("matrix", torch.eye(channels), persistent=active)

    def fit(self, x: Float[Tensor, "n c t"]) -> None:
        """Set the whitening from scaled training windows `x`, estimated on the CPU in double precision so a shared
        GPU holds no copy of the sample."""
        if not self.active:
            return
        values, vectors = torch.linalg.eigh(self.shrunk(x.cpu()))
        self.matrix.copy_(((vectors * values.rsqrt()) @ vectors.T).float())

    @staticmethod
    def shrunk(x: Float[Tensor, "n c t"]) -> Float[Tensor, "c c"]:
        """Mean over time points of each one's Ledoit-Wolf (2004) covariance across the windows."""
        samples = x.double().permute(2, 0, 1)
        samples = samples - samples.mean(1, keepdim=True)
        n = samples.shape[1]
        covariance = samples.transpose(1, 2) @ samples / n
        scale = covariance.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
        target = scale * torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
        spread = (samples.pow(2).sum(-1).pow(2).mean(-1) - covariance.pow(2).sum((1, 2))) / n
        shrinkage = (spread / (covariance - target).pow(2).sum((1, 2))).clamp(0, 1)[:, None, None]
        return ((1 - shrinkage) * covariance + shrinkage * target).mean(0)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c t"]:
        return torch.einsum("dc,bct->bdt", self.matrix, x) if self.active else x

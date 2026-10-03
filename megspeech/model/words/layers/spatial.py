"""Sensors merged by attention over their positions on the helmet, with spatial dropout (Défossez et al. 2023)."""

import math
from typing import override

import mne
import torch
from jaxtyping import Float
from torch import Tensor, nn

LAYOUT = "Vectorview-all"
FREQUENCIES = 32
"""Fourier frequencies per axis the attention scores are a sum over: Défossez et al.'s K."""
RADIUS = 0.2
"""Spatial dropout radius in the [0, 1]-normalised layout: Défossez et al.'s, about a fifth of the helmet."""


class SpatialAttention(nn.Module):
    """Each output channel a softmax-weighted sum of the sensors, its weights a learned function of their 2D position.

    The function is a Fourier series over the layout, so nearby sensors weigh alike. In training every sensor within
    `RADIUS` of one random point per window is dropped and the weights renormalised over the rest."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.xy: Tensor
        self.fourier: Tensor
        self.register_buffer("xy", self.positions(), persistent=False)
        self.register_buffer("fourier", self.features(self.xy), persistent=False)
        self.scores = nn.Parameter(torch.randn(channels, self.fourier.shape[1]) / FREQUENCIES)

    @staticmethod
    def positions() -> Float[Tensor, "c 2"]:
        """Each sensor's box centre in the MNE layout, min-max normalised to [0, 1], in the recordings' sorted order."""
        layout = mne.channels.read_layout(LAYOUT)
        centres = layout.pos[:, :2] + layout.pos[:, 2:] / 2
        order = sorted(range(len(layout.names)), key=lambda i: layout.names[i].replace(" ", ""))
        xy = torch.tensor(centres[order], dtype=torch.float32)
        return (xy - xy.amin(0)) / (xy.amax(0) - xy.amin(0))

    @staticmethod
    def features(xy: Float[Tensor, "c 2"]) -> Float[Tensor, "c f"]:
        """Cosine and sine of every frequency pair `k x + l y`, `k, l < FREQUENCIES`."""
        k = torch.arange(FREQUENCIES, dtype=torch.float32)
        phase = 2 * math.pi * (xy[:, 0, None, None] * k[:, None] + xy[:, 1, None, None] * k[None, :]).flatten(1)
        return torch.cat([phase.cos(), phase.sin()], dim=1)

    def kept(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c"]:
        """1 for each window's sensors outside `RADIUS` of its random point in training, else everywhere."""
        if not self.training:
            return x.new_ones(x.shape[:2])
        centre = torch.rand(len(x), 2, device=x.device)
        return (torch.cdist(centre, self.xy) > RADIUS).to(x.dtype)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b o t"]:
        logits = self.scores @ self.fourier.T
        weights = (logits - logits.amax(1, keepdim=True)).exp().to(x.dtype)
        kept = self.kept(x)
        merged = torch.einsum("oc,bct->bot", weights, x * kept[..., None])
        return merged / (kept @ weights.T)[..., None]

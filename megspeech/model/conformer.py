"""A residual Conformer block over hidden MEG features (Gulati et al. 2020), the encoder of MEGConformer, the
LibriBrain 2025 phoneme-track winner: half feed-forward, self-attention, depthwise convolution, half feed-forward."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.norm import MaskedBatchNorm1d

HEADS = 4
KERNEL = 25
"""Depthwise kernel in samples: 0.5 s at 50 Hz, MEGConformer's span."""
EXPANSION = 4


class Conformer(nn.Module):
    def __init__(self, width: int, dropout: float, heads: int = HEADS, kernel: int = KERNEL) -> None:
        super().__init__()
        self.first = self.feedforward(width, dropout)
        self.attend_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(width)
        self.pointwise = nn.Conv1d(width, 2 * width, 1)
        self.depthwise = nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width)
        self.batch_norm = MaskedBatchNorm1d(width)
        self.conv_out = nn.Sequential(nn.SiLU(), nn.Conv1d(width, width, 1), nn.Dropout(dropout))
        self.second = self.feedforward(width, dropout)
        self.out_norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def feedforward(width: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, EXPANSION * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(EXPANSION * width, width),
            nn.Dropout(dropout),
        )

    @override
    def forward(self, x: Float[Tensor, "b h t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b h t"]:
        """`valid` rows alone set the batch statistics; the rest are padding."""
        h = x.transpose(1, 2)
        h = h + self.first(h) / 2
        attended = self.attend_norm(h)
        h = h + self.drop(self.attention(attended, attended, attended, need_weights=False)[0])
        mixed = nn.functional.glu(self.pointwise(self.conv_norm(h).transpose(1, 2)), dim=1)
        h = h + self.conv_out(self.batch_norm(self.depthwise(mixed), valid)).transpose(1, 2)
        h = h + self.second(h) / 2
        return self.out_norm(h).transpose(1, 2)

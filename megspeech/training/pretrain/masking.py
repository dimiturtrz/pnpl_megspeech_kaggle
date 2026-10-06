"""What the student is not shown: time patches of its window and whole sensors."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor


@dataclass(frozen=True)
class Masking:
    ratio: float = 0.4
    """Share of the window's time patches the student does not see: data2vec masks 0.65 of speech frames, wav2vec 2.0
    0.5 of them."""
    patch: float = 0.05
    """Seconds one masked patch spans, about one 50 ms response component."""
    dropout: float = 0.1
    """Share of sensors zeroed in each window, a light channel dropout."""

    def __post_init__(self) -> None:
        if not 0 < self.ratio < 1:
            raise ValueError(f"mask ratio {self.ratio} is not in (0, 1)")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"channel dropout {self.dropout} is not in [0, 1)")

    def patches(self, rate: float) -> int:
        """Samples per patch at `rate`, at least one."""
        return max(1, round(self.patch * rate))

    def draw(self, rows: int, samples: int, rate: float, generator: torch.Generator) -> Bool[Tensor, "b t"]:
        """Each of `rows` windows' masked samples: `ratio` of its patches, exactly, chosen by rank of a uniform draw."""
        width = self.patches(rate)
        count = -(-samples // width)
        chosen = max(1, round(self.ratio * count))
        ranks = torch.rand(rows, count, generator=generator).argsort(dim=1).argsort(dim=1)
        return (ranks < chosen).repeat_interleave(width, dim=1)[:, :samples]

    def apply(
        self,
        x: Float[Tensor, "b c t"],
        hidden: Bool[Tensor, "b t"],
        token: Float[Tensor, "c 1"],
        generator: torch.Generator,
    ) -> Float[Tensor, "b c t"]:
        """`x` with the `hidden` samples replaced by the learned `token` and a share of sensors zeroed."""
        kept = torch.rand(x.shape[:2], generator=generator).ge(self.dropout).to(x.device)[..., None]
        return torch.where(hidden[:, None, :], token.to(x.dtype), x) * kept

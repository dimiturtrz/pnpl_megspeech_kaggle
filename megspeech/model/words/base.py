"""What the sentence model reads each word through: one window in, an embedding `width` wide and a word head out."""

from abc import ABC, abstractmethod
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.model.words.heads import WordHead

EPSILON = 1e-6


class WordEncoder(nn.Module, ABC):
    """A word classifier over one window; subclasses build `head` from `classifier` after their own layers, so a
    seeded initialisation draws in the order checkpoints were trained with."""

    head: nn.Module

    def __init__(self, width: int, *, instance: bool) -> None:
        super().__init__()
        self.width = width
        self.instance = instance

    def classifier(self, dropout: float, head: WordHead = WordHead.LINEAR) -> nn.Module:
        return head.build(self.width, dropout)

    @abstractmethod
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        """`valid` rows alone set any batch statistics; the rest are padding."""

    def positions(
        self, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> Float[Tensor, "g w e"]:
        """Each span position's embedding of its row of `x`, zeros where `slot == len(x)` marks it empty. `onset` is
        read only by encoders whose rows are spans, not words."""
        return self.placed(self.embed(x, self.occupied(x, slot)), slot)

    @staticmethod
    def occupied(x: Float[Tensor, "r c t"], slot: Int[Tensor, "g w"]) -> Bool[Tensor, " r"]:
        """Which rows of `x` some span position names; `slot == len(x)` names none."""
        rows = len(x)
        return torch.zeros(rows + 1, dtype=torch.bool, device=x.device).index_fill_(0, slot.flatten(), value=True)[
            :rows
        ]

    @staticmethod
    def placed(rows: Float[Tensor, "r *f"], slot: Int[Tensor, "g w"]) -> Float[Tensor, "g w *f"]:
        """Each span position's row of `rows`, zeros where `slot == len(rows)` marks it empty."""
        return nn.functional.pad(rows, (0, 0) * (rows.dim() - 1) + (0, 1))[slot]

    def time_width(self) -> int:
        """Channels of the per-timestep features `embed_timed` returns."""
        raise NotImplementedError(f"{type(self).__name__} keeps no per-timestep features")

    def embed_timed(
        self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, "b h s"]]:
        """`embed`, and the per-timestep features it pooled."""
        raise NotImplementedError(f"{type(self).__name__} keeps no per-timestep features")

    def positions_timed(
        self, x: Float[Tensor, "r c t"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w e"], Float[Tensor, "g w h s"]]:
        """`positions`, and each span position's per-timestep features, zeros where `slot == len(x)` marks it empty."""
        embedded, times = self.embed_timed(x, self.occupied(x, slot))
        return self.placed(embedded, slot), self.placed(times, slot)

    def tap_widths(self) -> tuple[int, ...]:
        """Channels of each feature map `embed_tapped` can read, in depth order."""
        raise NotImplementedError(f"{type(self).__name__} keeps no intermediate feature maps")

    def embed_tapped(
        self, x: Float[Tensor, "b c t"], taps: tuple[int, ...], levels: tuple[int, ...], valid: Bool[Tensor, " b"]
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, "b k"]]:
        """`embed`, and the feature maps at depths `taps` each mean-pooled over `levels` equal chunks of time."""
        raise NotImplementedError(f"{type(self).__name__} keeps no intermediate feature maps")

    def positions_tapped(
        self, x: Float[Tensor, "r c t"], slot: Int[Tensor, "g w"], taps: tuple[int, ...], levels: tuple[int, ...]
    ) -> tuple[Float[Tensor, "g w e"], Float[Tensor, "g w k"]]:
        """`positions`, and each span position's `embed_tapped` pyramid, zeros where `slot == len(x)` marks it
        empty."""
        embedded, pooled = self.embed_tapped(x, taps, levels, self.occupied(x, slot))
        return self.placed(embedded, slot), self.placed(pooled, slot)

    def core(self) -> tuple[nn.Module, ...]:
        """The modules `TransferConfig.encoder_lr` scales: the whole encoder, but for one over a pretrained trunk."""
        return (self,)

    def normalise(self, x: Float[Tensor, "b c t"], present: Bool[Tensor, "b 1 t"]) -> Float[Tensor, "b c t"]:
        """`x` standardised when `instance`, else unchanged."""
        return self.standardise(x, present) if self.instance else x

    @staticmethod
    def standardise(x: Float[Tensor, "b c t"], present: Bool[Tensor, "b 1 t"]) -> Float[Tensor, "b c t"]:
        """`x` z-scored per window and sensor over its `present` samples, zero elsewhere."""
        count = present.sum(-1, keepdim=True).clamp(min=1)
        centred = (x - (x * present).sum(-1, keepdim=True) / count) * present
        spread = (centred.square().sum(-1, keepdim=True) / count).sqrt()
        return centred / (spread + EPSILON)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b v"]:
        return self.head(self.embed(x))

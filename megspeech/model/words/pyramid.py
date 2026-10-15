"""An intermediate-depth ("pyramid") readout over a frozen trained model (HDND, arXiv 2609.24095): the encoder's
feature maps at chosen depths, mean-pooled over a temporal pyramid, fused with the span transformer's output into
one norm-bounded correction added to both heads' logits. Only the readout trains; the model under it is frozen."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.base import EPSILON, WordEncoder


@dataclass(frozen=True)
class PyramidConfig:
    taps: tuple[int, ...] = ()
    """Depths of the word encoder's feature maps the readout reads (`WordEncoder.tap_widths`, 0 the first block's);
    empty reads none and leaves the model as it is, trainable."""
    levels: tuple[int, ...] = (1, 2)
    """Equal chunks of the window each tapped map is mean-pooled over, one pooling per entry: the temporal pyramid."""
    width: int = 256
    """Channels the pooled pyramid is fused into before it meets the span transformer's output."""
    dropout: float = 0.1
    bound: float = 0.25
    """The correction's norm never exceeds this (`c` tanh-scaled); HDND chose 0.15 or 0.25 on val."""

    def __post_init__(self) -> None:
        if list(self.taps) != sorted(set(self.taps)) or any(tap < 0 for tap in self.taps):
            raise ValueError(f"pyramid taps {self.taps} must increase strictly from 0")
        if not self.levels or any(level < 1 for level in self.levels):
            raise ValueError(f"pyramid levels {self.levels} must be positive chunk counts")
        if self.bound <= 0:
            raise ValueError(f"pyramid bound {self.bound} must be positive")

    @property
    def active(self) -> bool:
        return bool(self.taps)

    def readout(self, words: WordEncoder, width: int) -> "Depth":
        """The readout this config names over `words`; `width` is that of the refined positions."""
        return Pyramid(self, words, width) if self.active else Surface()


PYRAMID = PyramidConfig()


class Depth(nn.Module, ABC):
    """How a `SentenceNet` reads its encoder and corrects its heads' logits."""

    @abstractmethod
    def positions(
        self, words: WordEncoder, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w e"], Float[Tensor, "g w k"]]:
        """Each span position's word embedding, and the features `correct` reads beside the refined positions."""

    @abstractmethod
    def correct(
        self,
        taps: Float[Tensor, "g w k"],
        refined: Float[Tensor, "g w h"],
        context: Float[Tensor, "g w v"],
        word: Float[Tensor, "g w v"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context and word heads' logits, corrected."""

    @abstractmethod
    def hold(self, net: nn.Module) -> None:
        """Whatever of `net` besides this readout must stay as trained, kept so after a mode change."""

    def fresh(self) -> set[str]:
        """The state keys, under `net`'s `pyramid.`, a warm start from a model without this readout lacks."""
        return {f"pyramid.{key}" for key in self.state_dict()}


class Surface(Depth):
    """The encoder's last map alone, the logits uncorrected and every weight trainable."""

    @override
    def positions(
        self, words: WordEncoder, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w e"], Float[Tensor, "g w k"]]:
        slots = words.positions(x, onset, slot)
        return slots, slots[..., :0]

    @override
    def correct(
        self,
        taps: Float[Tensor, "g w k"],
        refined: Float[Tensor, "g w h"],
        context: Float[Tensor, "g w v"],
        word: Float[Tensor, "g w v"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        return context, word

    @override
    def hold(self, net: nn.Module) -> None:
        """Nothing is held."""


class Pyramid(Depth):
    """The tapped maps' temporal pyramid and the refined position, fused into one residual on both heads' logits,
    its norm at most `bound`; the last layer starts at zero, so training starts from the frozen model's own logits."""

    def __init__(self, config: PyramidConfig, words: WordEncoder, width: int) -> None:
        """`width` is that of the refined positions the context head reads."""
        super().__init__()
        depths = words.tap_widths()
        if config.taps[-1] >= len(depths):
            raise ValueError(f"pyramid taps {config.taps} reach past the encoder's {len(depths)} feature maps")
        self.taps = config.taps
        self.levels = config.levels
        self.bound = config.bound
        pooled = sum(depths[tap] for tap in config.taps) * sum(config.levels)
        self.fuse = nn.Sequential(
            nn.LayerNorm(pooled), nn.Linear(pooled, config.width), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.out = nn.Linear(config.width + width, VOCAB_SIZE)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @override
    def positions(
        self, words: WordEncoder, x: Float[Tensor, "r c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w e"], Float[Tensor, "g w k"]]:
        return words.positions_tapped(x, slot, self.taps, self.levels)

    @override
    def correct(
        self,
        taps: Float[Tensor, "g w k"],
        refined: Float[Tensor, "g w h"],
        context: Float[Tensor, "g w v"],
        word: Float[Tensor, "g w v"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        residual = self.residual(taps, refined)
        return context + residual, word + residual

    def residual(self, taps: Float[Tensor, "g w k"], refined: Float[Tensor, "g w h"]) -> Float[Tensor, "g w v"]:
        """The correction: the fused features' raw logits rescaled to norm `bound * tanh(norm)`."""
        raw = self.out(torch.cat([self.fuse(taps), refined], dim=-1))
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
        return raw * (self.bound * torch.tanh(norm) / norm)

    @override
    def hold(self, net: nn.Module) -> None:
        """Every other module of `net` in evaluation mode with its gradients off: the frozen model's dropout,
        normalisation statistics and weights stay as trained."""
        for child in (child for child in net.children() if child is not self):
            child.train(mode=False)
            child.requires_grad_(requires_grad=False)

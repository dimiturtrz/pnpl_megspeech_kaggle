"""What trains beside the 50-way heads: nothing, or every heard word's embedding aligned to its t5 embedding."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.model.pretraining import Pretraining
from megspeech.model.sentence import SentenceNet
from megspeech.objective.prototypes import Lexicon
from megspeech.objective.semantic import Semantic


class Auxiliary(ABC):
    """The module a run's optimiser steps, and the loss it adds to the heads'."""

    @abstractmethod
    def trained(self, net: SentenceNet, compiled: Callable[[nn.Module], nn.Module]) -> nn.Module: ...

    @abstractmethod
    def loss(
        self, model: nn.Module, outputs: Sequence[Tensor], tokens: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""] | float:
        """`outputs` past the two heads', at span positions whose words are `tokens` (`Split.token`)."""

    @staticmethod
    def of(weight: float, root: DataRoot, device: torch.device) -> "Auxiliary":
        """Nothing at weight 0, so the lexicon is never read; the semantic alignment at `weight` otherwise."""
        return Aligned(weight, Lexicon.load(Lexicon.path(root)).to(device)) if weight else Silent()


class Silent(Auxiliary):
    @override
    def trained(self, net: SentenceNet, compiled: Callable[[nn.Module], nn.Module]) -> nn.Module:
        return net

    @override
    def loss(self, model: nn.Module, outputs: Sequence[Tensor], tokens: Int[Tensor, "g w"]) -> float:
        return 0.0


@dataclass(frozen=True)
class Aligned(Auxiliary):
    """Every heard word's window, in or out of the 50, trained toward its t5-large embedding (`Semantic`)."""

    weight: float
    lexicon: Lexicon

    @override
    def trained(self, net: SentenceNet, compiled: Callable[[nn.Module], nn.Module]) -> nn.Module:
        return compiled(Pretraining(net, Semantic(net.words.width).to(self.lexicon.vectors.device)))

    @override
    def loss(self, model: nn.Module, outputs: Sequence[Tensor], tokens: Int[Tensor, "g w"]) -> Float[Tensor, ""]:
        (projected,) = outputs
        kinds = self.lexicon.rows(tokens).flatten()
        semantic = cast("Pretraining", model).semantic
        return self.weight * semantic.loss(projected.flatten(0, 1), self.lexicon.vectors[kinds.clamp(min=0)], kinds)

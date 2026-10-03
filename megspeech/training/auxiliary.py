"""What trains beside the 50-way heads: nothing, or terms read off every heard word's embedding — its t5
embedding aligned, its first phoneme classified."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.model.pretraining import Pretraining
from megspeech.model.sentence import SentenceNet
from megspeech.objective.prototypes import Lexicon
from megspeech.training.config import OptimConfig
from megspeech.training.terms import Aligned, Lexical, Pronounced, Term


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
    def of(optim: OptimConfig, root: DataRoot, device: torch.device) -> "Auxiliary":
        """Nothing when every weight is 0, so no table is read; otherwise the terms of the nonzero weights."""
        terms: list[Term] = []
        if optim.semantic:
            terms.append(Aligned(optim.semantic, Lexicon.load(Lexicon.path(root)).to(device)))
        if optim.lexical:
            terms.append(Lexical(optim.lexical, Lexicon.load(Lexicon.path(root)).to(device)))
        if optim.phonetic:
            pronunciation = Pronunciation.load(Pronunciation.path(root)).to(device)
            terms.append(Pronounced(optim.phonetic, pronunciation, optim.positions))
        return Joint(tuple(terms)) if terms else Silent()


class Silent(Auxiliary):
    @override
    def trained(self, net: SentenceNet, compiled: Callable[[nn.Module], nn.Module]) -> nn.Module:
        return net

    @override
    def loss(self, model: nn.Module, outputs: Sequence[Tensor], tokens: Int[Tensor, "g w"]) -> float:
        return 0.0


@dataclass(frozen=True)
class Joint(Auxiliary):
    """The sentence model wrapped with one head per term (`Pretraining`), their losses summed."""

    terms: tuple[Term, ...]

    @override
    def trained(self, net: SentenceNet, compiled: Callable[[nn.Module], nn.Module]) -> nn.Module:
        return compiled(Pretraining(net, nn.ModuleList(term.head(net.words.width) for term in self.terms)))

    @override
    def loss(self, model: nn.Module, outputs: Sequence[Tensor], tokens: Int[Tensor, "g w"]) -> Float[Tensor, ""]:
        heads = cast("Pretraining", model).heads
        parts = zip(self.terms, heads, outputs, strict=True)
        return torch.stack([term.loss(head, output, tokens) for term, head, output in parts]).sum()

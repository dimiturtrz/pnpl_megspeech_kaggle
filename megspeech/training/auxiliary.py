"""What trains beside the 50-way heads: nothing, or terms read off every heard word's embedding — its t5
embedding aligned, its word classified, its leading phonemes classified."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import pandas as pd
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.labels.context_tags import ContextTags
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.syntax import PartsOfSpeech
from megspeech.model.auxiliary import Tap, WithAuxiliary
from megspeech.model.sentence import SentenceNet
from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.auxiliary.phonetic import PhoneticTerm
from megspeech.objective.auxiliary.semantic import SemanticTerm
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.auxiliary.syntactic import SyntacticTerm
from megspeech.objective.auxiliary.tagging import TagTerm
from megspeech.objective.targets.speech import Speech, Targets
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import AuxiliaryTerm
from megspeech.training.config import OptimConfig


@dataclass(frozen=True)
class Auxiliary(ABC):
    """The module a run's optimiser steps, and the terms its auxiliary heads add to the heads' loss."""

    terms: tuple[AuxiliaryTerm, ...]

    @abstractmethod
    def trained(
        self, net: SentenceNet, compiled: Callable[[WithAuxiliary], WithAuxiliary]
    ) -> SentenceNet | WithAuxiliary: ...

    @staticmethod
    def of(optim: OptimConfig, root: DataRoot, device: torch.device, words: pd.DataFrame) -> "Auxiliary":
        """Nothing when every weight is 0, so no table is read; otherwise the terms of the nonzero weights, the speech
        term's targets those of the training `words` (`Corpus.positions`)."""
        builders: tuple[tuple[float, Callable[[], AuxiliaryTerm]], ...] = (
            (
                optim.speech,
                lambda: SpeechTerm(
                    optim.speech,
                    Speech.gather(Targets.directory(root), words, centred=optim.speech_centred).to(device),
                ),
            ),
            (optim.semantic, lambda: SemanticTerm(optim.semantic, Lexicon.load(Lexicon.path(root)).to(device))),
            (optim.lexical, lambda: LexicalTerm(optim.lexical, Lexicon.load(Lexicon.path(root)).to(device))),
            (
                optim.phonetic,
                lambda: PhoneticTerm(
                    optim.phonetic, Pronunciation.load(Pronunciation.path(root)).to(device), optim.positions
                ),
            ),
            (
                optim.syntactic,
                lambda: SyntacticTerm(optim.syntactic, PartsOfSpeech.load(PartsOfSpeech.path(root)).to(device)),
            ),
            (
                optim.tagging,
                lambda: TagTerm(optim.tagging, ContextTags.load(ContextTags.path(root)).tensor(words, device)),
            ),
        )
        terms = tuple(build() for weight, build in builders if weight)
        return Joint(terms, optim.tap) if terms else NoAuxiliary()


@dataclass(frozen=True)
class NoAuxiliary(Auxiliary):
    """The sentence model trained alone."""

    terms: tuple[AuxiliaryTerm, ...] = ()

    @override
    def trained(self, net: SentenceNet, compiled: Callable[[WithAuxiliary], WithAuxiliary]) -> SentenceNet:
        return net


@dataclass(frozen=True)
class Joint(Auxiliary):
    """The sentence model wrapped with one head per term (`WithAuxiliary`), each reading its `tap`."""

    tap: Tap = Tap.WORD

    @override
    def trained(self, net: SentenceNet, compiled: Callable[[WithAuxiliary], WithAuxiliary]) -> WithAuxiliary:
        width = self.tap.width(net)
        heads = nn.ModuleList(term.readout(net.words.head, width) for term in self.terms)
        return compiled(WithAuxiliary(net, heads, self.tap))

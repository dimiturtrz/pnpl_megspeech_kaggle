"""The cross-subject instance term (megspeech-7gf.147.2): InfoNCE over each word's local encoder embedding, after the
subject layer, in which a word's positives are the same MASC token instance (`megspeech.data.labels.instances`) heard
by another subject, and its negatives every other word of the batch. What two brains share hearing one token is the
word; what differs is the subject, so pulling them together trains the encoder toward what a zero-shot subject shares.

Supervised-contrastive form
(`megspeech.objective.auxiliary.contrastive.supcon`). The same subject's other hearing of the token
(its other session) is neither positive nor negative: it is what a subject-specific code also matches.

Under `Apart` (`--optim.instance_apart`, megspeech-ct5) a word's other-word neighbours in its own sentence, whoever
heard it, are no negatives either: a sentence's words are ~0.25 s apart, so their 1 s windows share most of the
sound, and contrasting them would teach the encoder to tell one sound from itself (as `audio.py` leaves out a
window's own span). `Together` keeps every other word a negative.
"""

from dataclasses import dataclass
from typing import ClassVar, override

import pandas as pd
import torch
from jaxtyping import Bool, Int
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.labels.instances import NO_INSTANCE, Instances
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.auxiliary.contrastive.supcon import Positives, SupConTerm


@dataclass(frozen=True)
class Together:
    """No pair of the batch's slots is related beyond the term's own exclusions."""

    def related(self, at: Int[Tensor, " m"]) -> Bool[Tensor, "m m"]:
        """Which pairs of the slots holding words `at` are left out as negatives: none."""
        return torch.zeros(len(at), len(at), dtype=torch.bool, device=at.device)


@dataclass(frozen=True)
class Apart(Together):
    """Slots whose words share a `sentence` (`Instances.sentences`) are related."""

    sentence: Int[Tensor, " n"]

    @staticmethod
    def of(words: pd.DataFrame, device: torch.device) -> "Apart":
        """The sentences of training `words` (`Corpus.positions`)."""
        return Apart(torch.from_numpy(Instances.sentences(words)).to(device))

    @override
    def related(self, at: Int[Tensor, " m"]) -> Bool[Tensor, "m m"]:
        sentence = self.sentence[at]
        return sentence[:, None] == sentence[None, :]


@dataclass(frozen=True)
class InstancePositives(Positives):
    """A word's positives are the same MASC token heard by another subject."""

    instance: Int[Tensor, " n"]
    subject: Int[Tensor, " n"]
    neighbours: Together = Together()

    @override
    def pairs(self, target: Int[Tensor, " m"]) -> tuple[Bool[Tensor, "m m"], Bool[Tensor, "m m"]]:
        """Over the batch's flattened slots, which pairs are candidates (labelled, not the slot itself, not the same
        subject's hearing of the same token, not a `neighbours`-related other token) and which of those positives (the
        same token, another subject)."""
        valid = target != IGNORE
        at = target.clamp(min=0)
        instance = torch.where(valid, self.instance[at], NO_INSTANCE)
        subject = self.subject[at]
        known = instance != NO_INSTANCE
        same = (instance[:, None] == instance[None, :]) & known[:, None]
        mine = subject[:, None] == subject[None, :]
        both = valid[:, None] & valid[None, :]
        itself = torch.eye(len(target), dtype=torch.bool, device=target.device)
        candidate = both & ~(same & mine) & ~itself & ~(self.neighbours.related(at) & ~same)
        return candidate, candidate & same


@dataclass(frozen=True)
class InstanceTerm(SupConTerm):
    """InfoNCE over the batch's words with each training row's MASC `instance` and the `subject` who heard it."""

    local: ClassVar[bool] = True
    instance: Int[Tensor, " n"]
    subject: Int[Tensor, " n"]
    neighbours: Together = Together()

    @staticmethod
    def of(weight: float, words: pd.DataFrame, device: torch.device, *, apart: bool = False) -> "InstanceTerm":
        """The term over training `words` (`Corpus.positions`), `apart` leaving each word's sentence out of its
        negatives."""
        instances = Instances.of(words)
        if not instances.heard:
            raise ValueError("no training word is a MASC instance, so no word has a cross-subject positive")
        return InstanceTerm(
            weight=weight,
            instance=torch.from_numpy(instances.code).to(device),
            subject=torch.from_numpy(instances.subject).to(device),
            neighbours=Apart.of(words, device) if apart else Together(),
        )

    @override
    def positives(self) -> InstancePositives:
        return InstancePositives(self.instance, self.subject, self.neighbours)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != len(self.instance):
            raise ValueError(f"{len(self.instance)} instance rows for {len(split.y)} words")
        return torch.arange(len(self.instance), device=self.instance.device)

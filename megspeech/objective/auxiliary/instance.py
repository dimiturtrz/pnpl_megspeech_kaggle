"""The cross-subject instance term (megspeech-7gf.147.2): InfoNCE over each word's local encoder embedding, after the
subject layer, in which a word's positives are the same MASC token instance (`megspeech.data.labels.instances`) heard
by another subject, and its negatives every other word of the batch. What two brains share hearing one token is the
word; what differs is the subject, so pulling them together trains the encoder toward what a zero-shot subject shares.

Supervised-contrastive form (Khosla et al. 2020, `L_out`): an anchor's loss is the log-sum-exp over its candidates less
the mean logit of its positives, averaged over the anchors with any. The same subject's other hearing of the token (its
other session) is neither positive nor negative: it is what a subject-specific code also matches.

Under `Apart` (`--optim.instance_apart`, megspeech-ct5) a word's other-word neighbours in its own sentence, whoever
heard it, are no negatives either: a sentence's words are ~0.25 s apart, so their 1 s windows share most of the
sound, and contrasting them would teach the encoder to tell one sound from itself (as `audio.py` leaves out a
window's own span). `Together` keeps every other word a negative.

The logits are cosines times `exp(theta) * ln N` over the batch's `N` labelled words, `theta` learned from 0: at
`ln N` a positive at cosine 1 among `N - 1` negatives at cosine 0 holds half the softmax, where its gradient
`p (1 - p)` is largest, so the scale starts where the term learns fastest whatever the batch's size.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast, override

import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.batching.split import Split
from megspeech.data.labels.instances import NO_INSTANCE, Instances
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.term import AuxiliaryTerm, Reading


class Similarity(nn.Module):
    """The embeddings unchanged, the term read off them directly; holds the learned `theta` of the logit scale."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(()))

    @override
    def forward(self, x: Float[Tensor, "g w d"]) -> Float[Tensor, "g w d"]:
        return x

    def scale(self, candidates: Int[Tensor, ""]) -> Float[Tensor, ""]:
        """The logit scale over `candidates` words: `exp(theta) * ln candidates`, at least `ln 2`."""
        return self.theta.exp() * candidates.clamp(min=2).float().log()


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
class InstanceTerm(AuxiliaryTerm):
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
    def head(self, width: int) -> nn.Module:
        return Similarity()

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != len(self.instance):
            raise ValueError(f"{len(self.instance)} instance rows for {len(split.y)} words")
        return torch.arange(len(self.instance), device=self.instance.device)

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

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        flat = reading.target.flatten()
        candidate, positive = self.pairs(flat)
        output = reading.output
        with torch.autocast(output.device.type, enabled=False):
            unit = functional.normalize(output.flatten(0, 1).float(), dim=-1)
            logits = unit @ unit.T * cast("Similarity", reading.head).scale((flat != IGNORE).sum())
        open_ = candidate.any(dim=1, keepdim=True)
        spread = torch.where(open_, logits.masked_fill(~candidate, -math.inf), 0.0).logsumexp(dim=1)
        count = positive.sum(dim=1)
        pulled = (logits * positive).sum(dim=1) / count.clamp(min=1)
        return self.averaged(spread - pulled, count > 0, reading.rows.flatten())

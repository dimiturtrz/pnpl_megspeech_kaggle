"""The same-word contrastive term (megspeech-7gf.86): supervised contrastive loss
(`megspeech.objective.auxiliary.supcon`) over the embeddings the term's tap names, in which a word's positives are
every other labelled word of the batch with its 50-way label from a different sentence, and its negatives every
labelled word of another label. It pulls what the closed-vocabulary cross-entropy already pulls toward one class
weight into neighbourhoods a kNN readout finds (`megspeech.evaluation.studies.knnreadout`), invariant to the context
and timing that tell two hearings apart.

A pair in one sentence is neither positive nor negative: a sentence's words are ~0.25 s apart, so their 1 s windows
overlap and contrasting them would teach the encoder to tell one sound from itself (`Apart`'s precedent); the same
label in one sentence is left out for the same reason. A MASC sentence is one sentence whoever heard it
(`Instances.sentences`), so another subject's hearing of the anchor's own sentence is left out too.

The batch's target packs both a word's sentence and label (`SameWord.pack`), so the strategy reads neither a table
nor the split.
"""

from dataclasses import dataclass
from typing import ClassVar, override

import pandas as pd
import torch
from jaxtyping import Bool, Int
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.labels.instances import Instances
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.auxiliary.supcon import Positives, SupConTerm


@dataclass(frozen=True)
class SameWord(Positives):
    """Positives share a 50-way label and differ in sentence; same-sentence pairs are no candidates."""

    @staticmethod
    def pack(label: Int[Tensor, " n"], sentence: Int[Tensor, " n"]) -> Int[Tensor, " n"]:
        """Each word's `sentence * VOCAB_SIZE + label`, `IGNORE` where the word is outside the 50."""
        return torch.where(label == IGNORE, IGNORE, sentence * VOCAB_SIZE + label)

    @override
    def pairs(self, target: Int[Tensor, " m"]) -> tuple[Bool[Tensor, "m m"], Bool[Tensor, "m m"]]:
        valid = target != IGNORE
        code = target.clamp(min=0)
        label, sentence = code % VOCAB_SIZE, code // VOCAB_SIZE
        itself = torch.eye(len(target), dtype=torch.bool, device=target.device)
        candidate = valid[:, None] & valid[None, :] & ~itself & (sentence[:, None] != sentence[None, :])
        return candidate, candidate & (label[:, None] == label[None, :])


@dataclass(frozen=True)
class SameWordTerm(SupConTerm):
    """Supervised contrastive loss over the batch's labelled words, with each training row's `sentence`."""

    local: ClassVar[bool] = False
    sentence: Int[Tensor, " n"]

    @staticmethod
    def of(weight: float, words: pd.DataFrame, device: torch.device) -> "SameWordTerm":
        """The term over training `words` (`Corpus.positions`)."""
        return SameWordTerm(weight=weight, sentence=torch.from_numpy(Instances.sentences(words)).to(device))

    @override
    def positives(self) -> SameWord:
        return SameWord()

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != len(self.sentence):
            raise ValueError(f"{len(self.sentence)} sentence rows for {len(split.y)} words")
        return SameWord.pack(split.y, self.sentence)

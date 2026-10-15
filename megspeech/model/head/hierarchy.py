"""The context head's 50 words read flat, or factorised through the part of speech its own tag head reads.

Factorised, `log p(w|x) = log sum_t p(t|x) p(w|t,x)`: `p(t|x)` is a tag head over `CLASSES` on the refined position,
trained on every heard word (`TagTerm`, in or out of the 50: ~2.5x the 50-way labels); `p(w|t,x)` is the context
head's own logits softmaxed over the words tag `t` admits. Which words a tag admits is read off the training words'
in-context tags (`ContextTags`), as `p(t|w)`.

The marginal is soft, not hard. A hard map (each word at its majority tag) makes `log p(w) = log p(tag(w)|x) +
log p(w|tag(w),x)`, and a word said in two roles loses one: "to" is a particle 58% and an adposition 42% of the time,
"that" a subordinator, determiner and pronoun, so a confident correct ADP reading would score "to" from the wrong
group. Summing over every tag a word admits keeps each reading's evidence; it costs only a broader competition within
a tag. A tag admits a word when its share of that word's tokens clears `ADMIT`, below which a reading is not told
apart from the tagger's own errors. The 50-way label conditions on the word being one of the 50, so the mixture is
renormalised over them: the mass of tags admitting none of the 50 (`OTHER`) drops out.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.labels.context_tags import CLASSES
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.model.config import ContextConfig
from megspeech.objective.auxiliary.classified.tagging import Tagging

LOG = logging.getLogger(__name__)

TAGGER_ACCURACY = 0.98
"""Universal-tag accuracy spaCy reports for `en_core_web_trf` (its model card), the tagger `ContextTags` ran."""
ADMIT = 2 * (1 - TAGGER_ACCURACY)
"""The least share of a word's tokens a tag needs to admit it: twice the tagger's error rate."""


class Hierarchy(nn.Module, ABC):
    """What turns the context head's logits into its scores, given the refined positions it read."""

    @staticmethod
    def of(context: ContextConfig) -> "Hierarchy":
        builders: dict[bool, Callable[[], Hierarchy]] = {
            False: Flat,
            True: lambda: Factorised(context.width, context.dropout),
        }
        return builders[context.hierarchical]()

    @staticmethod
    def untagged() -> Int[Tensor, " n"]:
        """The tags of a trainer that loaded none."""
        raise ValueError("a factorised context head reads the training words' tags: build the trainer with Trainer.of")

    @abstractmethod
    @override
    def forward(self, logits: Float[Tensor, "g w v"], refined: Float[Tensor, "g w h"]) -> Float[Tensor, "g w v"]: ...

    @abstractmethod
    def tagging(self, word_head: nn.Module) -> nn.Module:
        """The module whose tag head the tag auxiliary trains: the word head's, unless this holds one."""

    @abstractmethod
    def admit(self, y: Int[Tensor, " n"], tagged: Callable[[], Int[Tensor, " n"]]) -> None:
        """Fix which words each tag admits from the training words' labels `y` and their tags, read by `tagged`."""


class Flat(Hierarchy):
    """The logits as they are: no parameters, no state, the model as it was before the hierarchy existed."""

    @override
    def forward(self, logits: Float[Tensor, "g w v"], refined: Float[Tensor, "g w h"]) -> Float[Tensor, "g w v"]:
        return logits

    @override
    def tagging(self, word_head: nn.Module) -> nn.Module:
        return word_head

    @override
    def admit(self, y: Int[Tensor, " n"], tagged: Callable[[], Int[Tensor, " n"]]) -> None:
        return


class Factorised(Hierarchy, Tagging):
    """The soft factorisation of the module docstring; `admitted[t, w]` says tag `t` admits word `w`."""

    reads_context = True
    admitted: Bool[Tensor, "t v"]

    def __init__(self, width: int, dropout: float, tags: int = len(CLASSES)) -> None:
        super().__init__()
        self.tag = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, tags))
        self.register_buffer("admitted", torch.ones(tags, VOCAB_SIZE, dtype=torch.bool))

    @override
    def forward(self, logits: Float[Tensor, "g w v"], refined: Float[Tensor, "g w h"]) -> Float[Tensor, "g w v"]:
        prior = self.tag(refined).float().log_softmax(-1)
        return (prior[..., None] + self.within(logits.float())).logsumexp(-2).log_softmax(-1)

    def within(self, logits: Float[Tensor, "*b v"]) -> Float[Tensor, "*b t v"]:
        """`log p(w|t)`: `logits` softmaxed over each tag's admitted words, -inf at the rest."""
        allowed = self.admitted
        masked = logits[..., None, :].masked_fill(~allowed, torch.finfo(logits.dtype).min)
        return masked.log_softmax(-1).masked_fill(~allowed, -torch.inf)

    @override
    def readout(self) -> nn.Module:
        return self.tag

    @override
    def tagging(self, word_head: nn.Module) -> nn.Module:
        return self

    @staticmethod
    def affinity(y: Int[Tensor, " n"], tags: Int[Tensor, " n"], count: int) -> Float[Tensor, "t v"]:
        """`p(t|w)`: each word's share of tokens at each of `count` tags, over the labelled, tagged rows; a word with
        none is a zero column."""
        known = (y != IGNORE) & (tags != UNKNOWN)
        cells = torch.bincount(tags[known] * VOCAB_SIZE + y[known], minlength=count * VOCAB_SIZE)
        counts = cells.view(count, VOCAB_SIZE).float()
        return counts / counts.sum(0).clamp(min=1)

    @override
    def admit(self, y: Int[Tensor, " n"], tagged: Callable[[], Int[Tensor, " n"]]) -> None:
        """Each word admitted by every tag holding at least `ADMIT` of its tokens; a word never tagged, by all."""
        tags = tagged()
        if len(tags) != len(y):
            raise ValueError(f"{len(tags)} tags for {len(y)} words")
        shares = self.affinity(y, tags, len(self.admitted))
        admitted = (shares >= ADMIT) | (shares.sum(0) == 0)
        self.admitted.copy_(admitted)
        LOG.info(
            "hierarchy: %d of %d words tagged, %d admitted by more than one tag",
            int((shares.sum(0) > 0).sum()),
            VOCAB_SIZE,
            int((admitted.sum(0) > 1).sum()),
        )

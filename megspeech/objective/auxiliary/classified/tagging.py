"""The tag auxiliary: each heard word's embedding classified into the part of speech it was said as in its sentence."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import override

from jaxtyping import Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.labels.context_tags import CLASSES
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.term import Classified


class Tagging(nn.Module):
    """A word head that holds a tag head of its own."""

    reads_context: bool = False
    """Whether the tag head reads the span-refined position rather than the word embedding."""

    @abstractmethod
    def readout(self) -> nn.Module:
        """The tag head: embeddings in, logits over `CLASSES` out."""


@dataclass(frozen=True)
class TagTerm(Classified):
    """Every training word's embedding, in or out of the 50, classified into the `CLASSES` tag it carries in its
    sentence (`ContextTags`); `tags` holds each training word's, in split order. The head is the word head's own tag
    head, so this loss is what trains the tags its readouts are mixed by."""

    weight: float
    tags: Int[Tensor, " n"]

    @override
    def head(self, width: int) -> nn.Module:
        raise ValueError("the tag auxiliary reads the word head's own tag head: set context.conv.head to tagged")

    @override
    def readout(self, word_head: nn.Module, width: int) -> nn.Module:
        if not isinstance(word_head, Tagging):
            return self.head(width)
        return word_head.readout()

    @override
    def count(self) -> int:
        return len(CLASSES)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        if len(split.y) != len(self.tags):
            raise ValueError(f"{len(self.tags)} tags for {len(split.y)} words")
        return self.tags

    @override
    def classify(self, token: Int[Tensor, "g w"]) -> Int[Tensor, "g w"]:
        return token.where(token != IGNORE, UNKNOWN)

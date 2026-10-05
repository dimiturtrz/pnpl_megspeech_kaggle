"""A word head factorised through the part of speech the embedding says the word has."""

from typing import override

from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.labels.context_tags import CLASSES
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.objective.auxiliary.tagging import Tagging


class TaggedHead(Tagging):
    """A word head factorised through a part of speech: each tag has its own linear readout `h_t` of the 50 words,
    and word `w` scores `logsumexp_t(log_softmax(tag)_t + h_t(x)_w)`. The tag head is trained beside it
    (`TagTerm`), so the readout that fires for a word is the one of the tag the embedding says it has."""

    def __init__(self, width: int, dropout: float, tags: int = len(CLASSES)) -> None:
        super().__init__()
        self.tag = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, tags))
        self.words = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, tags * VOCAB_SIZE))

    @override
    def forward(self, x: Float[Tensor, "*b e"]) -> Float[Tensor, "*b v"]:
        prior = self.tag(x).float().log_softmax(-1)
        per_tag = self.words(x).float().unflatten(-1, (prior.shape[-1], VOCAB_SIZE))
        return (prior[..., None] + per_tag).logsumexp(-2)

    @override
    def readout(self) -> nn.Module:
        """The tag head: embeddings in, logits over `CLASSES` out."""
        return self.tag

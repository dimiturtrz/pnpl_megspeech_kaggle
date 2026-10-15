"""A word head: word embeddings in, logits over the 50 words out."""

from jaxtyping import Float
from torch import Tensor, nn


class WordReadout(nn.Module):
    """A word head reading the word embeddings alone; one that reads the span-refined positions too overrides `attach`
    and `read`."""

    def attach(self, context_width: int) -> None:
        """Size the head for the `context_width` refined positions it reads: none, by default."""

    def read(self, slots: Float[Tensor, "g w e"], refined: Float[Tensor, "g w h"]) -> Float[Tensor, "g w v"]:
        """The words over the embeddings `slots`, given the `refined` positions it may read."""
        return self(slots)

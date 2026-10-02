"""The semantic auxiliary: each heard word's embedding aligned to its t5 embedding by in-batch SigLIP."""

from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.objective.heads import BIAS, DIM, TEMPERATURE


class Semantic(nn.Module):
    """A projection of word embeddings into t5-large's space, scored against the batch's distinct words."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.project = nn.Linear(width, DIM)
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE))
        self.bias = nn.Parameter(torch.tensor(BIAS))

    @override
    def forward(self, embedded: Float[Tensor, "*b w"]) -> Float[Tensor, "*b d"]:
        return self.project(embedded)

    def loss(
        self, projected: Float[Tensor, "n d"], targets: Float[Tensor, "n d"], kinds: Int[Tensor, " n"]
    ) -> Float[Tensor, ""]:
        """SigLIP of each row against every distinct kind in the batch, positive on its own kind only.

        A negative kind is no word (padding, or a word without a target): it is neither a row nor a candidate.
        Candidates are deduplicated by kind without a host sync, so the step keeps fixed shapes.
        """
        valid = kinds >= 0
        same = kinds[:, None] == kinds[None, :]
        candidate = valid & ~same.tril(-1).any(dim=1)
        similarity = functional.normalize(projected.float(), dim=-1) @ functional.normalize(targets.float(), dim=-1).T
        logits = self.temperature.exp() * similarity + self.bias
        pairs = functional.binary_cross_entropy_with_logits(logits, same.float(), reduction="none")
        return ((pairs * candidate).sum(-1) * valid).sum() / valid.sum().clamp(min=1)

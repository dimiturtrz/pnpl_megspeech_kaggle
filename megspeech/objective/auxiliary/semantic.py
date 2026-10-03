"""The semantic auxiliary: each heard word's embedding aligned to its t5 embedding by in-batch SigLIP."""

from dataclasses import dataclass
from typing import cast, override

from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.objective.scorer import SigmoidScorer
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import AuxiliaryTerm


class Semantic(SigmoidScorer):
    """A projection of word embeddings into t5-large's space, scored against the batch's distinct words."""

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
        logits = self.score(projected, functional.normalize(targets.float(), dim=-1))
        pairs = functional.binary_cross_entropy_with_logits(logits, same.float(), reduction="none")
        return ((pairs * candidate).sum(-1) * valid).sum() / valid.sum().clamp(min=1)


@dataclass(frozen=True)
class SemanticTerm(AuxiliaryTerm):
    """Every heard word's window, in or out of the 50, trained toward its t5-large embedding (`Semantic`)."""

    weight: float
    lexicon: Lexicon

    @override
    def head(self, width: int) -> nn.Module:
        return Semantic(width).to(self.lexicon.vectors.device)

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        kinds = self.lexicon.rows(target).flatten()
        return cast("Semantic", head).loss(output.flatten(0, 1), self.lexicon.vectors[kinds.clamp(min=0)], kinds)

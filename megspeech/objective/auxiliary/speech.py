"""The speech auxiliary: each training word's embedding aligned to the wav2vec2 latent of the audio it was heard in,
by in-batch SigLIP over the batch's words, each word its own kind."""

from dataclasses import dataclass
from typing import cast, override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.objective.auxiliary.semantic import Semantic
from megspeech.objective.targets.speech import Speech
from megspeech.objective.term import AuxiliaryTerm


@dataclass(frozen=True)
class SpeechTerm(AuxiliaryTerm):
    """Every training word with a speech target (`Speech`) trained toward it; the rest are left out."""

    weight: float
    speech: Speech
    softmax: bool = False
    """Contrast by softmax over the batch's kinds (`Semantic.softmax_loss`) rather than independent sigmoids."""

    @override
    def head(self, width: int) -> nn.Module:
        return Semantic(width, self.speech.vectors.shape[1]).to(self.speech.vectors.device)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        """The row of each of `split`'s words among the targets: the training split's own words, in order."""
        if len(split.y) != len(self.speech.rows):
            raise ValueError(f"{len(split.y)} words against {len(self.speech.rows)} speech rows")
        return self.speech.rows

    @override
    def unweighted(
        self, head: nn.Module, output: Float[Tensor, "g w _"], target: Int[Tensor, "g w"]
    ) -> Float[Tensor, ""]:
        kinds = target.flatten()
        semantic = cast("Semantic", head)
        contrast = semantic.softmax_loss if self.softmax else semantic.loss
        return contrast(output.flatten(0, 1), self.speech.vectors[kinds.clamp(min=0)], kinds)

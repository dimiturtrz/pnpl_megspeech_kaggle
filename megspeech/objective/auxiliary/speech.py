"""The speech auxiliary: each training word's embedding aligned to the wav2vec2 latent of the audio it was heard in,
by in-batch SigLIP over the batch's words, each word its own kind."""

from dataclasses import dataclass
from typing import cast, override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.objective.auxiliary.semantic import Semantic
from megspeech.objective.targets.speech import Speech
from megspeech.objective.term import AuxiliaryTerm, Reading


@dataclass(frozen=True)
class SpeechTerm(AuxiliaryTerm):
    """Every training word with a speech target (`Speech`) trained toward it; the rest are left out."""

    weight: float
    speech: Speech
    softmax: bool = False
    """Contrast by softmax over the batch's kinds (`Semantic.softmax_loss`) rather than independent sigmoids."""

    @override
    def head(self, width: int) -> nn.Module:
        return Semantic(width, self.speech.vectors.shape[1]).to(self.speech.rows.device)

    @override
    def labels(self, split: Split) -> Int[Tensor, " n"]:
        """The row of each of `split`'s words among the targets: the training split's own words, in order."""
        if len(split.y) != len(self.speech.rows):
            raise ValueError(f"{len(split.y)} words against {len(self.speech.rows)} speech rows")
        return self.speech.rows

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        semantic = cast("Semantic", reading.head)
        contrast = semantic.softmax_loss if self.softmax else semantic.loss
        return reading.against(contrast, self.speech.targets(reading.target.flatten()), reading.target)

"""The context head over the fixed per-word prototypes."""

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.config import ContextConfig
from megspeech.model.hierarchy import Hierarchy
from megspeech.objective.anchored import AnchoredHead
from megspeech.objective.scorer import DIM


class ContextReadout:
    """Builds the context head a config names."""

    @staticmethod
    def hierarchy(context: ContextConfig) -> Hierarchy:
        """What turns the context head's logits into its scores: flat, or factorised through the part of speech."""
        return Hierarchy.of(context)

    @staticmethod
    def build(context: ContextConfig, prototypes: Float[Tensor, "v d"] | None) -> nn.Module:
        """The context head over the fixed per-word `prototypes` (None stands in zeros for a checkpoint that will
        overwrite them): cosine to them as anchors when the config names a source, else the objective's own."""
        if not context.anchor.active:
            return context.objective.head(context.width, prototypes)
        placeholder = torch.zeros(VOCAB_SIZE, context.anchor_dim or DIM) if prototypes is None else prototypes
        return AnchoredHead(context.width, placeholder, trainable=context.anchor_train)

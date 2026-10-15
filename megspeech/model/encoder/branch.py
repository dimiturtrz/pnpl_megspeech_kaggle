"""A saved run's stage-1 encoder as one source of fused features (megspeech-7gf.131)."""

from collections.abc import Sequence
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput
from megspeech.device import Device
from megspeech.model.config import ContextConfig
from megspeech.model.encoder.encoding import Encoding
from megspeech.model.input.stem import Stem
from megspeech.model.input.subjects import Unseen

SCALE_MEDIAN = "scale.median"
"""A saved state's per-channel scaler median: one entry per channel the encoder reads."""


class Branch(Encoding):
    """A saved run's stage-1 encoder, and the projection of its word features to the transformer's width."""

    def __init__(self, name: str, context: ContextConfig, channels: int, width: int, unseen: Unseen) -> None:
        """A branch called `name` over `channels` rows, its weights to be loaded."""
        super().__init__(Stem.placeholder(channels), context, channels, unseen)
        self.name = name
        self.project = nn.Linear(self.words.width, width)

    @classmethod
    def load(cls, directory: Path, own: ContextConfig, channels: int, unseen: Unseen = Unseen.IDENTITY) -> "Branch":
        """The run in `directory`'s encoder, shaped by its own config and holding its weights, projecting to `own`'s
        width; its input must be the model's own channels."""
        state = torch.load(directory / "model.pt", map_location=Device.CPU, weights_only=True)
        context = ContextConfig.load(directory)
        reads = state[SCALE_MEDIAN].shape[0]
        if reads != channels:
            raise ValueError(f"stem {directory.name} reads {reads} channels; the model reads {channels}")
        branch = cls(directory.name, context, channels, own.width, unseen)
        branch.adopt(state)
        return branch

    @classmethod
    def many(
        cls, directories: Sequence[Path], own: ContextConfig, channels: int, unseen: Unseen = Unseen.IDENTITY
    ) -> list["Branch"]:
        """The runs in `directories`, each loaded as `load` does."""
        return [cls.load(directory, own, channels, unseen) for directory in directories]

    def projected(self, spans: SpanInput) -> Float[Tensor, "g w d"]:
        """Each span position's features through this encoder, projected to the transformer's width."""
        return self.project(self.words.positions(self.prepare(spans.x, spans.subject), spans.onset, spans.slot))

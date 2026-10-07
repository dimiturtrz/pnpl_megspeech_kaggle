"""A saved run's stage-1 encoder as one source of fused features (megspeech-7gf.131)."""

from collections.abc import Sequence
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.config import ContextConfig
from megspeech.model.encoding import Encoding
from megspeech.model.input.stem import Stem
from megspeech.model.input.subjects import Unseen
from megspeech.model.words.layers.mean_max import MeanMax

SCALE_MEDIAN = "scale.median"
"""A saved state's per-channel scaler median: one entry per channel the encoder reads."""


class Branch(Encoding):
    """A saved run's stage-1 encoder, and the projection of its word features to the transformer's width."""

    def __init__(self, name: str, context: ContextConfig, channels: int, width: int, unseen: Unseen) -> None:
        """A branch called `name` over `channels` rows, its weights to be loaded."""
        super().__init__(Stem.placeholder(channels), context, channels, unseen)
        self.name = name
        self.project = nn.Linear(self.words.width // context.tokens, width)

    @classmethod
    def load(cls, directory: Path, own: ContextConfig, channels: int, unseen: Unseen = Unseen.IDENTITY) -> "Branch":
        """The run in `directory`'s encoder, shaped by its own config and holding its weights, projecting to `own`'s
        width; its input must be the model's own: the same channels and tokens per word."""
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        context = ContextConfig.load(directory)
        reads = state[SCALE_MEDIAN].shape[0]
        if reads != channels or context.tokens != own.tokens:
            raise ValueError(
                f"stem {directory.name} reads {reads} channels and {context.tokens} tokens per word; the model "
                f"reads {channels} and {own.tokens}"
            )
        branch = cls(directory.name, context, channels, own.width, unseen)
        branch.adopt(state)
        return branch

    @classmethod
    def many(
        cls, directories: Sequence[Path], own: ContextConfig, channels: int, unseen: Unseen = Unseen.IDENTITY
    ) -> list["Branch"]:
        """The runs in `directories`, each loaded as `load` does."""
        return [cls.load(directory, own, channels, unseen) for directory in directories]

    def projected(
        self,
        x: Float[Tensor, "r c t"],
        subject: Int[Tensor, " r"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        tokens: int,
    ) -> Float[Tensor, "g w s d"]:
        """Each span position's `tokens` features through this encoder, projected to the transformer's width; `slot`
        names a position's row of `x`, and `len(x)` an empty one."""
        slots = self.words.positions(self.prepare(x, subject), onset, slot)
        return self.project(MeanMax.segmented(slots, tokens))

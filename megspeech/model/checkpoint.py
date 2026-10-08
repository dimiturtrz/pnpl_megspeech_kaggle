"""Trained sentence-context models read back from their directories."""

from pathlib import Path

import torch

from megspeech.device import Device
from megspeech.model.branch import SCALE_MEDIAN, Branch
from megspeech.model.config import ContextConfig
from megspeech.model.fusion import Fusion
from megspeech.model.input.stem import Stem
from megspeech.model.input.subjects import Unseen
from megspeech.model.sentence import SentenceNet


class Checkpoint:
    """A saved `SentenceNet`."""

    @staticmethod
    def load(path: Path, unseen: Unseen = Unseen.IDENTITY) -> SentenceNet:
        """The checkpoint at `path` in the shape its directory's config records; its state overwrites a placeholder
        scaler shaped as the state's own: MEG channels or subspace components, or a cached encoder's feature
        channels."""
        state = torch.load(path, map_location=Device.CPU)
        channels = state[SCALE_MEDIAN].shape[0]
        context = ContextConfig.load(path.parent)
        branches = Branch.many([path.parent.parent / name for name in context.fused], context, channels, unseen)
        model = SentenceNet(Stem.placeholder(channels), context=context, unseen=unseen, channels=channels).fuse(
            Fusion(branches, context.branch_drop)
        )
        model.load_state_dict(state)
        return model

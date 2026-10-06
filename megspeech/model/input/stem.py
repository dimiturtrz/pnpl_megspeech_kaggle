"""The input stages a window passes before the word encoder: scaler, whitener, subject layer."""

import torch
from torch import nn

from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import SubjectLayer, Unseen
from megspeech.model.input.whitener import Whitener


class Stem:
    @staticmethod
    def of(
        scale: nn.Module, channels: int, *, whiten: bool, subjects: int, unseen: Unseen
    ) -> tuple[nn.Module, Whitener, nn.Module]:
        """`scale`, a whitener over `channels` (inactive unless `whiten`) and a layer for `subjects` subjects reading
        a subject it has no matrix for as `unseen`."""
        return scale, Whitener(channels, active=whiten), SubjectLayer.of(subjects, channels, unseen)

    @staticmethod
    def placeholder(channels: int) -> Scaler:
        """A scaler of `channels` rows whose state is yet to be loaded."""
        return Scaler(torch.zeros(1, channels, 1))

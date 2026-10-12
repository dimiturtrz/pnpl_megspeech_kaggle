"""The input stages a window passes before the word encoder: subject levels, scaler, whitener, subject layer."""

from dataclasses import dataclass

import torch
from torch import nn

from megspeech.model.input.levels import Levels
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import SubjectLayer, Unseen
from megspeech.model.input.whitener import Whitener


@dataclass(frozen=True)
class Stages:
    """Which optional input stages are on: per-subject `levels`, `whiten`ing, and a layer for `subjects` subjects
    (none at 0)."""

    levels: bool = False
    whiten: bool = False
    subjects: int = 0


class Stem:
    @staticmethod
    def of(
        scale: nn.Module, channels: int, stages: Stages, unseen: Unseen
    ) -> tuple[Levels, nn.Module, Whitener, nn.Module]:
        """Per-subject levels over `channels`, `scale`, a whitener and a subject layer reading a subject it has no
        matrix for as `unseen`, each optional stage as `stages` sets it."""
        built = Levels(channels, active=stages.levels), scale, Whitener(channels, active=stages.whiten)
        return *built, SubjectLayer.of(stages.subjects, channels, unseen)

    @staticmethod
    def placeholder(channels: int) -> Scaler:
        """A scaler of `channels` rows whose state is yet to be loaded."""
        return Scaler(torch.zeros(1, channels, 1))

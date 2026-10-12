import torch
from torch import nn

from megspeech.model.input.levels import Levels
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.stem import Stages, Stem
from megspeech.model.input.subjects import SubjectLayer, Unseen
from megspeech.model.input.whitener import Whitener

CHANNELS = 3


def test_of() -> None:
    scale = nn.Identity()
    levels, kept, whitener, subjects = Stem.of(
        scale, CHANNELS, Stages(levels=True, whiten=True, subjects=2), Unseen.IDENTITY
    )
    assert isinstance(levels, Levels)
    assert levels.active
    assert kept is scale
    assert isinstance(whitener, Whitener)
    assert isinstance(subjects, SubjectLayer)
    assert not isinstance(Stem.of(scale, CHANNELS, Stages(), Unseen.IDENTITY)[3], SubjectLayer)


def test_placeholder() -> None:
    placeholder = Stem.placeholder(CHANNELS)
    assert isinstance(placeholder, Scaler)
    assert placeholder.median.shape == (CHANNELS, 1)
    assert torch.equal(placeholder.median, torch.zeros(CHANNELS, 1))

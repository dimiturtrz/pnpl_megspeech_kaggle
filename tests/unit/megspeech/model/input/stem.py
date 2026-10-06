import torch
from torch import nn

from megspeech.model.input.scaler import Scaler
from megspeech.model.input.stem import Stem
from megspeech.model.input.subjects import SubjectLayer, Unseen
from megspeech.model.input.whitener import Whitener

CHANNELS = 3


def test_of() -> None:
    scale = nn.Identity()
    kept, whitener, subjects = Stem.of(scale, CHANNELS, whiten=True, subjects=2, unseen=Unseen.IDENTITY)
    assert kept is scale
    assert isinstance(whitener, Whitener)
    assert isinstance(subjects, SubjectLayer)
    assert not isinstance(Stem.of(scale, CHANNELS, whiten=False, subjects=0, unseen=Unseen.IDENTITY)[2], SubjectLayer)


def test_placeholder() -> None:
    placeholder = Stem.placeholder(CHANNELS)
    assert isinstance(placeholder, Scaler)
    assert placeholder.median.shape == (CHANNELS, 1)
    assert torch.equal(placeholder.median, torch.zeros(CHANNELS, 1))

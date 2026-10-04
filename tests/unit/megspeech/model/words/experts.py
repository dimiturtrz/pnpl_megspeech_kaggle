import torch

from megspeech.data.constants import CHANNELS
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.expert import Crop
from megspeech.model.words.experts import ExpertWords
from tests.unit.megspeech.model.words.expert import window

SMALL = ConvConfig(hidden=8, dilations=(1,))


def test_embed() -> None:
    x = window()
    words = ExpertWords(SMALL, CHANNELS, (Crop(0, 10), Crop(5, 10))).eval()
    joined = torch.cat([branch.embed(x) for branch in words.branches()], dim=-1)
    assert words.width == 2 * SMALL.width
    assert torch.allclose(words.embed(x), words.features(joined[..., None])[..., 0])
    assert words(x).shape[0] == 3


def test_branches() -> None:
    words = ExpertWords(SMALL, CHANNELS, (Crop(0, 10), Crop(5, 10)))
    assert [branch.crop for branch in words.branches()] == [Crop(0, 10), Crop(5, 10)]

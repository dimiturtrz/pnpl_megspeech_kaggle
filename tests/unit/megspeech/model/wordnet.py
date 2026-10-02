import torch

from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import WORDS
from megspeech.model.wordnet import ConvConfig, WordNet

SMALL = ConvConfig(hidden=8, dilations=(1, 2))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_embed() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert WordNet(SMALL).embed(x).shape == (3, 2 * SMALL.hidden)


def test_word_net_forward() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert WordNet(SMALL).forward(x).shape == (3, WORDS)

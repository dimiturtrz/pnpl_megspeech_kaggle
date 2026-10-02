import torch

from megspeech.data.vocabulary import WORDS
from megspeech.model.cached import Cached
from megspeech.model.wordnet import ConvConfig

FEATURES = torch.arange(2.0 * 3 * 4).reshape(2, 3, 4)
WIDTH = 3 * 4


def test_embed():
    assert torch.equal(Cached(ConvConfig(), WIDTH).embed(FEATURES), FEATURES.reshape(2, WIDTH))


def test_normalise():
    assert Cached(ConvConfig(), WIDTH).normalise(FEATURES, torch.ones(2, 1, 4, dtype=torch.bool)) is FEATURES


def test_forward():
    assert Cached(ConvConfig(), WIDTH).forward(FEATURES).shape == (2, WORDS)

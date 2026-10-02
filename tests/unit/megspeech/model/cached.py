import torch

from megspeech.data.vocabulary import WORDS
from megspeech.model.cached import Cached
from megspeech.model.megxl_words import POOLED
from megspeech.model.wordnet import ConvConfig

FEATURES = torch.arange(2.0 * POOLED).reshape(2, POOLED, 1)


def test_embed():
    assert torch.equal(Cached(ConvConfig()).embed(FEATURES), FEATURES[..., 0])


def test_normalise():
    assert Cached(ConvConfig()).normalise(FEATURES, torch.ones(2, 1, 1, dtype=torch.bool)) is FEATURES


def test_forward():
    assert Cached(ConvConfig()).forward(FEATURES).shape == (2, WORDS)

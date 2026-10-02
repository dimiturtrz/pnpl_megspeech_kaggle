import pytest
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


def test_embed_segments_one_is_mean_and_max() -> None:
    net = WordNet(SMALL).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    h = net.spatial(x)
    for block in net.blocks:
        h = block(h)
    assert torch.allclose(net.embed(x), torch.cat([h.mean(-1), h.amax(-1)], dim=-1), atol=1e-6)


def test_embed_kernels_and_segments() -> None:
    config = ConvConfig(hidden=8, dilations=(1, 2), kernels=(3, 5, 7), segments=4)
    net = WordNet(config)
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert net.width == 2 * 8 * 4
    assert net.embed(x).shape == (3, net.width)
    assert net.forward(x).shape == (3, WORDS)


def test_conv_config_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        ConvConfig(kernels=(3, 4))

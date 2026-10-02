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


def test_embed_ssm_appends_one_layer() -> None:
    net = WordNet(ConvConfig(hidden=8, dilations=(1, 2), ssm=4))
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert len(net.blocks) == 3
    assert net.embed(x).shape == (3, 2 * 8)
    assert len(WordNet(SMALL).blocks) == 2


def test_embed_conformer_replaces_the_convolutions() -> None:
    net = WordNet(ConvConfig(hidden=8, dilations=(), conformer=2))
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert len(net.blocks) == 2
    assert net.embed(x).shape == (3, 2 * 8)


def test_normalise() -> None:
    x = 3.0 + 2.0 * torch.randn(4, 5, SAMPLES, generator=GENERATOR)
    present = torch.ones(4, 1, SAMPLES, dtype=torch.bool)
    present[0, :, 40:] = False
    assert torch.equal(WordNet(SMALL).normalise(x, present), x)
    normalised = WordNet(ConvConfig(hidden=8, dilations=(1, 2), instance=True)).normalise(x, present)
    assert torch.all(normalised[0, :, 40:] == 0)
    kept = normalised[0, :, :40]
    assert torch.allclose(kept.mean(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(kept.std(-1, unbiased=False), torch.ones(5), atol=1e-4)
    assert torch.allclose(normalised[1:].std(-1, unbiased=False), torch.ones(3, 5), atol=1e-4)


def test_normalise_flat_sensor_is_zero() -> None:
    net = WordNet(ConvConfig(hidden=8, dilations=(1, 2), instance=True))
    flat = net.normalise(torch.full((1, 2, SAMPLES), 7.0), torch.ones(1, 1, SAMPLES, dtype=torch.bool))
    assert torch.all(flat == 0)


def test_conv_config_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        ConvConfig(kernels=(3, 4))

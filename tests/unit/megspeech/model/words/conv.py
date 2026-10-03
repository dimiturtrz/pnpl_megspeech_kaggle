import pytest
import torch

from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.conv import ConvConfig, ConvWords

SMALL = ConvConfig(hidden=8, dilations=(1, 2))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_embed() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert ConvWords(SMALL).embed(x).shape == (3, 2 * SMALL.hidden)


def test_embed_segments_one_is_mean_and_max() -> None:
    net = ConvWords(SMALL).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    h = net.spatial(x)
    for block in net.blocks:
        h = block(h)
    assert torch.allclose(net.embed(x), torch.cat([h.mean(-1), h.amax(-1)], dim=-1), atol=1e-6)


def test_embed_kernels_and_segments() -> None:
    config = ConvConfig(hidden=8, dilations=(1, 2), kernels=(3, 5, 7), segments=4)
    net = ConvWords(config)
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert net.width == 2 * 8 * 4
    assert net.embed(x).shape == (3, net.width)
    assert net.forward(x).shape == (3, VOCAB_SIZE)


def test_embed_ssm_appends_one_layer() -> None:
    net = ConvWords(ConvConfig(hidden=8, dilations=(1, 2), ssm=4))
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert len(net.blocks) == 3
    assert net.embed(x).shape == (3, 2 * 8)
    assert len(ConvWords(SMALL).blocks) == 2


def test_embed_conformer_replaces_the_convolutions() -> None:
    net = ConvWords(ConvConfig(hidden=8, dilations=(), conformer=2))
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert len(net.blocks) == 2
    assert net.embed(x).shape == (3, 2 * 8)


def test_conv_config_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        ConvConfig(kernels=(3, 4))


def test_conv_config_rejects_lora() -> None:
    with pytest.raises(ValueError, match="retired"):
        ConvConfig(lora=4)

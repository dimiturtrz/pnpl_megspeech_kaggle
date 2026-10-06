from typing import cast

import pytest
import torch
from torch import nn

from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.conv import ConvConfig, ConvWords
from megspeech.model.words.layers.sensorwise import Sensorwise
from megspeech.model.words.layers.spatial import SpatialAttention

SMALL = ConvConfig(hidden=8, dilations=(1, 2))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_embed() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert ConvWords(SMALL).embed(x).shape == (3, 2 * SMALL.hidden)


def test_taps() -> None:
    words = ConvWords(SMALL).eval()
    x = torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)
    maps = words.taps(x)
    assert len(maps) == len(SMALL.dilations)
    assert all(m.shape == (2, SMALL.hidden, SAMPLES) for m in maps)
    assert torch.equal(words.pool(maps[-1]), words.embed(x))


def test_channels() -> None:
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    assert ConvWords(SMALL, 5).embed(x).shape == (3, 2 * SMALL.hidden)
    with pytest.raises(ValueError, match="spatial merger"):
        ConvWords(ConvConfig(hidden=8, dilations=(1,), merger=5), 5)


def test_width() -> None:
    assert SMALL.width == 2 * SMALL.hidden
    assert ConvConfig(hidden=8, segments=3).width == 48
    assert ConvConfig(hidden=8, segments=3, queries=2).width == 16
    assert ConvWords(ConvConfig(hidden=8, queries=2)).width == 16


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


def test_embed_attention_queries() -> None:
    net = ConvWords(ConvConfig(hidden=8, dilations=(1, 2), queries=3))
    assert net.width == 24
    assert net.embed(torch.randn(3, CHANNELS, 50, generator=GENERATOR)).shape == (3, 24)


def test_embed_merger_attends_before_the_convolution() -> None:
    net = ConvWords(ConvConfig(hidden=8, dilations=(1, 2), merger=5)).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    attention, convolution = cast("torch.nn.Sequential", net.spatial)
    assert isinstance(attention, SpatialAttention)
    assert cast("torch.nn.Conv1d", convolution).in_channels == 5
    assert net.embed(x).shape == (3, 2 * 8)
    assert isinstance(ConvWords(SMALL).spatial, torch.nn.Conv1d)


def test_sensorwise() -> None:
    config = ConvConfig(hidden=8, dilations=(1, 2), sensorwise=4)
    net = ConvWords(config, 5)
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    assert net.embed(x, torch.tensor([True, True, False])).shape == (3, 2 * config.hidden)
    with pytest.raises(ValueError, match="two spatial front ends"):
        ConvConfig(merger=5, sensorwise=4)


def test_front() -> None:
    assert isinstance(ConvWords.front(ConvConfig(hidden=8, sensorwise=4), 5), Sensorwise)
    assert isinstance(ConvWords.front(SMALL, 5), torch.nn.Conv1d)


def test_sensorwise_depth() -> None:
    config = ConvConfig(hidden=8, dilations=(1, 2, 4), sensorwise=4, sensorwise_depth=2)
    front = cast(Sensorwise, ConvWords.front(config, 5))
    assert [cast(nn.Conv1d, cast(nn.Sequential, block.body)[0]).dilation for block in front.stack] == [(1,), (2,)]
    full = cast(Sensorwise, ConvWords.front(ConvConfig(hidden=8, dilations=(1, 2, 4), sensorwise=4), 5))
    assert len(full.stack) == 3
    with pytest.raises(ValueError, match="sensorwise_depth"):
        ConvConfig(dilations=(1, 2), sensorwise=4, sensorwise_depth=3)


def test_sensorwise_site() -> None:
    config = ConvConfig(hidden=8, dilations=(1, 2), sensorwise=4, sensorwise_site=3)
    assert cast(Sensorwise, ConvWords.front(config, 6)).site == 3
    with pytest.raises(ValueError, match="needs sensorwise"):
        ConvConfig(sensorwise_site=3)


def test_sensorwise_out() -> None:
    config = ConvConfig(hidden=8, dilations=(1, 2), sensorwise=4, sensorwise_out=2)
    assert cast(Sensorwise, ConvWords.front(config, 6)).fuse.in_channels == 12
    with pytest.raises(ValueError, match="sensorwise_out"):
        ConvConfig(sensorwise_out=2)

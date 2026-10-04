import json
from pathlib import Path

import pytest
import torch
from torch import nn

from megspeech.model.config import CONTEXT as DEFAULT
from megspeech.model.config import FILE, ContextConfig, Encoder, OnsetEncoding
from megspeech.model.context.timing import Clock, Intervals
from megspeech.model.words.conv import ConvConfig, ConvWords
from megspeech.model.words.expert import Crop
from megspeech.model.words.experts import ExpertWords
from megspeech.model.words.megxl import POOLED, RESOLVED
from megspeech.objective.context import Objective

CONTEXT = ContextConfig(width=16, timescales=4)


def test_save(tmp_path: Path) -> None:
    assert CONTEXT.save(tmp_path).exists()


def test_load(tmp_path: Path) -> None:
    assert ContextConfig.load(tmp_path) == DEFAULT
    CONTEXT.save(tmp_path)
    assert ContextConfig.load(tmp_path) == CONTEXT
    ordered = ContextConfig(clock=Clock.ORDER)
    ordered.save(tmp_path)
    assert ContextConfig.load(tmp_path).clock is Clock.ORDER
    retrieval = ContextConfig(objective=Objective.SIGLIP)
    retrieval.save(tmp_path)
    assert ContextConfig.load(tmp_path).objective is Objective.SIGLIP
    timed = ContextConfig(intervals=Intervals.LOG)
    timed.save(tmp_path)
    assert ContextConfig.load(tmp_path).intervals is Intervals.LOG


def test_load_conv(tmp_path: Path) -> None:
    shaped = ContextConfig(conv=ConvConfig(kernels=(3, 5, 7), segments=4))
    shaped.save(tmp_path)
    assert ContextConfig.load(tmp_path) == shaped
    (tmp_path / FILE).write_text(json.dumps({"width": 16, "clock": "order"}))
    assert ContextConfig.load(tmp_path) == ContextConfig(width=16, clock=Clock.ORDER)


def test_combination() -> None:
    assert ContextConfig(encoder=Encoder.CACHED).subjects == 0
    assert ContextConfig(subjects=3).subjects == 3
    for encoder in (Encoder.CACHED, Encoder.RESOLVED):
        with pytest.raises(ValueError, match="subjects=3"):
            ContextConfig(encoder=encoder, subjects=3)
    with pytest.raises(ValueError, match="latency"):
        ContextConfig(latency=True, encoder=Encoder.CACHED)
    with pytest.raises(ValueError, match="latency"):
        ContextConfig(latency=True, conv=ConvConfig(queries=2))
    with pytest.raises(ValueError, match="whiten"):
        ContextConfig(whiten=True, encoder=Encoder.TRAINED)


def test_tokens() -> None:
    four = ConvConfig(segments=4)
    assert ContextConfig(conv=four).tokens == 1
    assert ContextConfig(conv=four, latency=True).tokens == 4


def test_build() -> None:
    conv = ConvConfig(hidden=8, dilations=(1,))
    config = ContextConfig(conv=conv)
    assert isinstance(Encoder.CONV.build(config), ConvWords)
    assert Encoder.CACHED.build(config).width == POOLED
    assert Encoder.RESOLVED.build(config).width == RESOLVED
    assert Encoder.TRAINED.build(config).width == conv.width
    narrow = Encoder.CONV.build(config, 5)
    assert isinstance(narrow, ConvWords)
    assert isinstance(narrow.spatial, nn.Conv1d)
    assert narrow.spatial.in_channels == 5
    assert Encoder.CACHED.build(config, 5).width == POOLED
    experts = ContextConfig(conv=conv, encoder=Encoder.EXPERTS, starts=(0, 10), lengths=(50, 10))
    assert isinstance(Encoder.EXPERTS.build(experts, 5), ExpertWords)
    assert Encoder.EXPERTS.build(experts, 5).width == 2 * conv.width


def test_crops() -> None:
    config = ContextConfig(encoder=Encoder.EXPERTS, starts=(0, 10), lengths=(50, 10))
    assert config.crops == (Crop(0, 50), Crop(10, 10))
    assert CONTEXT.crops == ()
    with pytest.raises(ValueError, match="one crop per expert"):
        ContextConfig(encoder=Encoder.EXPERTS)
    with pytest.raises(ValueError, match="one crop per expert"):
        ContextConfig(starts=(0,), lengths=(10,))
    with pytest.raises(ValueError, match="one crop per expert"):
        ContextConfig(encoder=Encoder.EXPERTS, starts=(0,), lengths=(10, 5))


def test_cached() -> None:
    assert [encoder for encoder in Encoder if encoder.cached] == [Encoder.CACHED, Encoder.RESOLVED, Encoder.TRAINED]


def test_onset_encoding_forward() -> None:
    timing = ContextConfig(width=16, timescales=4)
    onset = torch.tensor([[0.0, 0.3], [0.0, 0.0]])
    out = OnsetEncoding(timing).forward(onset, Intervals.NONE.features(onset, onset.ge(0)))
    assert out.shape == (2, 2, timing.width)
    assert torch.equal(out[0, 0], out[1, 1])
    timed = ContextConfig(width=16, timescales=4, intervals=Intervals.LOG)
    mask = torch.ones(2, 2, dtype=torch.bool)
    assert OnsetEncoding(timed).forward(onset, Intervals.LOG.features(onset, mask)).shape == (2, 2, 16)

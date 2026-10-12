import json
from pathlib import Path

import pytest
import torch

from core.tree import Tree
from megspeech.model.config import CONTEXT as DEFAULT
from megspeech.model.config import FILE, ContextConfig, Covariance, Encoder, OnsetEncoding
from megspeech.model.context.intervals import Intervals
from megspeech.model.context.timing import Clock
from megspeech.model.pyramid import PyramidConfig
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.heads import WordHead
from megspeech.model.words.spanned import SpannedConfig
from megspeech.objective.anchors import Source
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
    shaped = ContextConfig(conv=ConvConfig(segments=4))
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
    with pytest.raises(ValueError, match="whiten"):
        ContextConfig(whiten=True, encoder=Encoder.TRAINED)
    with pytest.raises(ValueError, match="siglip"):
        ContextConfig(anchor=Source.AUDIO, objective=Objective.SIGLIP)
    assert ContextConfig(hierarchical=True).hierarchical
    with pytest.raises(ValueError, match="factorised"):
        ContextConfig(hierarchical=True, objective=Objective.TOPK)
    with pytest.raises(ValueError, match="factorised"):
        ContextConfig(hierarchical=True, conv=ConvConfig(head=WordHead.TAGGED))
    assert ContextConfig(anchor=Source.TEXT).anchor is Source.TEXT
    for bad in (-1, 50):
        with pytest.raises(ValueError, match="anchor_dim"):
            ContextConfig(anchor=Source.AUDIO, anchor_dim=bad)
    with pytest.raises(ValueError, match="anchor_dim"):
        ContextConfig(anchor_dim=8)
    assert ContextConfig(anchor=Source.AUDIO, anchor_dim=49).anchor_dim == 49


def test_latency_retired() -> None:
    assert Tree(ContextConfig).load({"latency": False}) == ContextConfig()
    with pytest.raises(ValueError, match="7gf.67"):
        Tree(ContextConfig).load({"latency": True})


def test_crops() -> None:
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


def test_noise_covariance_needs_whiten() -> None:
    with pytest.raises(ValueError, match="noise covariance"):
        ContextConfig(covariance=Covariance.NOISE)
    assert ContextConfig(whiten=True, covariance=Covariance.NOISE).covariance is Covariance.NOISE


def test_branch_drop_is_a_probability() -> None:
    with pytest.raises(ValueError, match="branch_drop"):
        ContextConfig(branch_drop=1.0)


def test_pyramid_needs_an_unfused_conv_encoder(tmp_path: Path) -> None:
    tapped = ContextConfig(conv=ConvConfig(pyramid=PyramidConfig(taps=(1, 3), bound=0.15)))
    assert ContextConfig.load(tapped.save(tmp_path).parent) == tapped
    with pytest.raises(ValueError, match="conv encoder"):
        ContextConfig(encoder=Encoder.CACHED, conv=tapped.conv)
    with pytest.raises(ValueError, match="fused"):
        ContextConfig(fused=("other",), conv=tapped.conv)


def test_rate() -> None:
    """Only the spanned encoder's MEG-XL trunk fixes the cache rate it reads."""
    assert ContextConfig(encoder=Encoder.SPANNED).rate == SpannedConfig.rate
    assert ContextConfig().rate is None

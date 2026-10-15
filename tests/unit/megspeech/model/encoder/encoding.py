from dataclasses import replace

import pytest
import torch
from torch import nn

from megspeech.data.constants import CHANNELS
from megspeech.model.config import ContextConfig
from megspeech.model.encoder.encoding import Encoding
from megspeech.model.encoder.motion import Frozen
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import Unseen
from megspeech.model.words.conv import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def encoding() -> Encoding:
    return Encoding(nn.Identity(), CONTEXT, CHANNELS, Unseen.IDENTITY).eval()


def test_embed() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert encoding().embed(x, torch.zeros(3, dtype=torch.long)).shape == (3, CONTEXT.conv.width)


def test_prepare() -> None:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    assert encoding().prepare(x, torch.zeros(3, dtype=torch.long)).shape == x.shape


def test_encoder() -> None:
    made = encoding()
    assert {id(p) for p in made.encoder()} == {id(p) for p in made.parameters()}


def test_adopt() -> None:
    scale = Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR))
    donor = Encoding(scale, CONTEXT, CHANNELS, Unseen.IDENTITY)
    made = Encoding(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), CONTEXT, CHANNELS, Unseen.IDENTITY)
    made.adopt(donor.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(made.encoder(), donor.encoder(), strict=True))
    with pytest.raises(KeyError, match="lacks encoder entries"):
        made.adopt({})


def test_adopt_stem_without_subject_layer() -> None:
    """A state without `subjects.*` leaves a model's subject layer at identity, so subject 0 reads as the stem does."""
    donor = Encoding(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), CONTEXT, CHANNELS, Unseen.IDENTITY)
    plain = replace(CONTEXT, subjects=0)
    stem = Encoding(donor.scale, plain, CHANNELS, Unseen.IDENTITY).eval()
    stem.adopt(donor.state_dict())
    made = Encoding(donor.scale, replace(CONTEXT, subjects=33), CHANNELS, Unseen.IDENTITY).eval()
    made.adopt(stem.state_dict())
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    zero = torch.zeros(3, dtype=torch.long)
    assert torch.equal(made.embed(x, zero), stem.embed(x, zero))
    with pytest.raises(KeyError, match="lacks encoder entries"):
        made.adopt({})


def test_stage() -> None:
    made = encoding()
    assert made.stage() == (made.scale, made.whiten, made.subjects, made.words)


def test_train() -> None:
    made = encoding()
    made.motion = Frozen()
    assert made.train() is made
    assert made.training
    assert not any(module.training for module in made.stage())


def test_move() -> None:
    made = encoding().train()
    made.move(Frozen())
    assert isinstance(made.motion, Frozen)
    assert not made.words.training
    assert not any(p.requires_grad for p in made.encoder())
    made.train()
    assert not made.words.training

from dataclasses import replace
from typing import cast

import torch
from torch import nn

from megspeech.model.config import ContextConfig
from megspeech.model.context.span import SpanConformer
from megspeech.model.mixer import Mixer
from megspeech.model.words.conv import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
GENERATOR = torch.Generator().manual_seed(0)


def test_of() -> None:
    assert isinstance(Mixer.of(CONTEXT), nn.TransformerEncoder)
    assert isinstance(Mixer.of(replace(CONTEXT, mixing=3)), SpanConformer)
    alone = Mixer.of(replace(CONTEXT, layers=0))
    h = torch.randn(2, 3, CONTEXT.width, generator=GENERATOR)
    assert torch.equal(alone(h, src_key_padding_mask=torch.zeros(2, 3, dtype=torch.bool)), h)
    assert not list(alone.parameters())
    assert cast("nn.TransformerEncoder", Mixer.of(CONTEXT)).norm is None
    assert isinstance(cast("nn.TransformerEncoder", Mixer.of(replace(CONTEXT, final_norm=True))).norm, nn.LayerNorm)

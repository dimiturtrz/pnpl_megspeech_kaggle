from typing import cast

import pytest
import torch

from megspeech.model.context.droplayer import DropPathLayer
from megspeech.model.context.span import SpanConformer

GENERATOR = torch.Generator().manual_seed(0)
PADDING = torch.tensor([[False, False, True], [False, True, True]])


def test_layer() -> None:
    layer = SpanConformer.layer(8, 2, 0.1)
    assert layer.norm_first
    assert layer.linear1.out_features == 4 * 8


def test_forward() -> None:
    model = SpanConformer(8, 2, 2, 0.0, 3).eval()
    h = torch.randn(2, 3, 8, generator=GENERATOR)
    out = model.forward(h, PADDING)
    assert out.shape == h.shape
    changed = h.clone()
    changed[PADDING] = 100.0
    assert torch.allclose(model.forward(changed, PADDING)[~PADDING], out[~PADDING], atol=1e-5)


def test_ramp() -> None:
    model = SpanConformer(8, 2, 2, 0.0, 3)
    model.ramp(0.2)
    assert [p.rate for p in model.paths] == pytest.approx([0.1, 0.2])
    assert [cast("DropPathLayer", a).path.rate for a in model.attend] == pytest.approx([0.1, 0.2])
    h = torch.randn(2, 3, 8, generator=GENERATOR)
    assert torch.equal(model.eval().forward(h, PADDING), model.eval().forward(h, PADDING))
    assert model.train().forward(h, PADDING).shape == h.shape

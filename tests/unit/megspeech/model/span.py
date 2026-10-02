import torch

from megspeech.model.span import SpanConformer

GENERATOR = torch.Generator().manual_seed(0)
PADDING = torch.tensor([[False, False, True], [False, True, True]])


def test_forward() -> None:
    model = SpanConformer(8, 2, 2, 0.0, 3).eval()
    h = torch.randn(2, 3, 8, generator=GENERATOR)
    out = model.forward(h, PADDING)
    assert out.shape == h.shape
    changed = h.clone()
    changed[PADDING] = 100.0
    assert torch.allclose(model.forward(changed, PADDING)[~PADDING], out[~PADDING], atol=1e-5)

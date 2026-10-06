import torch
from torch import nn

from megspeech.model.context.droplayer import DropPathLayer

GENERATOR = torch.Generator().manual_seed(0)
PADDING = torch.tensor([[False, False, True], [False, True, True]])


def test_forward() -> None:
    """Rate 0 is the plain layer's output bit for bit; evaluation ignores the rate; a rate of 1 - eps keeps the
    skip path only."""
    layer = DropPathLayer(8, 2, 16, 0.0)
    plain = nn.TransformerEncoderLayer(8, 2, 16, 0.0, batch_first=True, norm_first=True)
    plain.load_state_dict(layer.state_dict(), strict=False)
    h = torch.randn(2, 3, 8, generator=GENERATOR)
    expected = plain.train().forward(h, src_key_padding_mask=PADDING)
    assert torch.equal(layer.train().forward(h, src_key_padding_mask=PADDING), expected)
    layer.path.rate = 0.5
    assert torch.equal(
        layer.eval().forward(h, src_key_padding_mask=PADDING), plain.eval().forward(h, src_key_padding_mask=PADDING)
    )
    layer.path.rate = 1.0 - 1e-9
    assert torch.allclose(layer.train().forward(h, src_key_padding_mask=PADDING), h)

import torch

from megspeech.model.context.mixing import Mixing

GENERATOR = torch.Generator().manual_seed(0)
PADDING = torch.tensor([[False, False, True], [False, True, True]])


def test_forward() -> None:
    """A word's neighbour reaches it through the kernel; padded words pass through unchanged."""
    mixing = Mixing(8, 0.0, 3).eval()
    h = torch.randn(2, 3, 8, generator=GENERATOR)
    out = mixing.forward(h, PADDING)
    assert torch.equal(out[PADDING], h[PADDING])
    nudged = h.clone()
    nudged[0, 1] += torch.randn(8, generator=GENERATOR)
    assert not torch.allclose(mixing.forward(nudged, PADDING)[0, 0], out[0, 0])

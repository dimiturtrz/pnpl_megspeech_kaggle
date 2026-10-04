import torch

from megspeech.model.words.layers.mean_max import MeanMax

GENERATOR = torch.Generator().manual_seed(0)


def test_mean_max_forward() -> None:
    h = torch.randn(3, 8, 50, generator=GENERATOR)
    assert torch.equal(MeanMax(1).forward(h), torch.cat([h.mean(-1), h.amax(-1)], dim=-1))
    assert MeanMax(4).forward(h).shape == (3, 2 * 8 * 4)


def test_segmented() -> None:
    """Each chunk's row is its own mean then max; one chunk is the pooled vector itself."""
    h = torch.randn(3, 8, 50, generator=GENERATOR)
    chunks = MeanMax.segmented(MeanMax(2).forward(h), 2)
    assert chunks.shape == (3, 2, 16)
    assert torch.allclose(chunks[:, 1], torch.cat([h[..., 25:].mean(-1), h[..., 25:].amax(-1)], dim=-1))
    pooled = MeanMax(1).forward(h)
    assert torch.equal(MeanMax.segmented(pooled, 1)[:, 0], pooled)

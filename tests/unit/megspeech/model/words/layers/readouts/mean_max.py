import torch

from megspeech.model.words.layers.readouts.mean_max import MeanMax

GENERATOR = torch.Generator().manual_seed(0)


def test_mean_max_forward() -> None:
    h = torch.randn(3, 8, 50, generator=GENERATOR)
    assert torch.equal(MeanMax(1).forward(h), torch.cat([h.mean(-1), h.amax(-1)], dim=-1))
    assert MeanMax(4).forward(h).shape == (3, 2 * 8 * 4)

import torch

from megspeech.model.words.layers.sites.sitedrop import SiteDrop

SITES = 6
DIM = 4
SAMPLES = 10


def test_forward() -> None:
    drop = SiteDrop(0.5).train()
    keep = drop.forward(torch.zeros(2000, SITES, DIM, SAMPLES))
    assert keep is not None
    assert keep.shape == (2000, SITES)
    assert keep.any(1).all()
    assert 0.4 < keep.float().mean() < 0.6
    heavy = SiteDrop(0.99).train().forward(torch.zeros(50, SITES, DIM, SAMPLES))
    assert heavy is not None
    assert heavy.any(1).all()
    assert drop.eval().forward(torch.zeros(2, SITES, DIM, SAMPLES)) is None
    assert SiteDrop(0.0).train().forward(torch.zeros(2, SITES, DIM, SAMPLES)) is None

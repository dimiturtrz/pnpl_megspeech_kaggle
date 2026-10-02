import torch

from megspeech.training.augment import Augment


def test_call() -> None:
    torch.manual_seed(0)
    x = torch.ones(64, 30, 5)
    assert torch.equal(Augment()(x), x)
    dropped = Augment(dropout=0.5)(x)
    assert 0.3 < (dropped[..., 0] == 0).float().mean() < 0.7
    assert torch.all(dropped.std(dim=-1) == 0)
    scaled = Augment(gain=0.2)(x)[..., 0]
    assert torch.all(scaled > 0)
    assert 0.1 < scaled.log().std() < 0.3

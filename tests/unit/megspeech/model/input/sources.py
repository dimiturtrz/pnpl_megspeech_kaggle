import torch

from megspeech.data.raw.runs import DEEP
from megspeech.model.input.sources import Sources

GENERATOR = torch.Generator().manual_seed(0)


def test_means() -> None:
    weight = torch.randn(36, 3, 3, generator=GENERATOR)
    means = Sources.means(weight)
    assert torch.allclose(means[1], weight[1:13].mean(dim=0), atol=1e-6)
    assert torch.allclose(means[12], means[1], atol=1e-6)
    assert torch.allclose(means[13], weight[13:23].mean(dim=0), atol=1e-6)
    assert torch.allclose(means[32], weight[23:33].mean(dim=0), atol=1e-6)
    assert torch.equal(means[DEEP], weight[DEEP])
    assert torch.allclose(means[34], weight[34])

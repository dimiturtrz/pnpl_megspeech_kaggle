import pytest
import torch

from megspeech.data.batching.windows import Windows
from megspeech.evaluation.studies.signal.topography import Evoked

SIGNAL = torch.tensor([[1.0, 1.0], [0.0, 3.0]])


def evoked() -> Evoked:
    """Four words: sensor 0 always 1 then 1, sensor 1 swinging 0 +- 1 then 3 +- 1."""
    noise = torch.tensor([1.0, -1.0, 1.0, -1.0])[:, None, None] * torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    return Evoked.of(Windows.stacked(SIGNAL + noise, centred=False), chunk=3)


def test_of() -> None:
    result = evoked()
    assert torch.allclose(result.mean, SIGNAL)
    assert torch.allclose(result.spread, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert result.count == 4


def test_t() -> None:
    t = evoked().t()
    assert torch.allclose(t[1], torch.tensor([0.0, 6.0]))
    assert t[0].gt(1e6).all()


def test_peaks() -> None:
    assert evoked().peaks(0, 1)[1] == pytest.approx(0.0)
    assert evoked().peaks(0, 2)[1] == pytest.approx(6.0)

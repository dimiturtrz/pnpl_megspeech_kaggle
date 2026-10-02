import torch

from megspeech.model.shared import Shared
from megspeech.model.subjects import SubjectLayer

GENERATOR = torch.Generator().manual_seed(0)


def test_subject_layer_forward() -> None:
    """Identity at initialisation; a change to one subject's matrix moves only that subject's rows."""
    layer = SubjectLayer(3, 4)
    x = torch.randn(5, 4, 6, generator=GENERATOR)
    subject = torch.tensor([0, 1, 2, 1, 0])
    assert torch.equal(layer.forward(x, subject), x)
    with torch.no_grad():
        layer.weight[1].mul_(2.0)
    moved = layer.forward(x, subject)
    assert torch.equal(moved[subject != 1], x[subject != 1])
    assert torch.allclose(moved[subject == 1], 2 * x[subject == 1])


def test_of() -> None:
    assert isinstance(SubjectLayer.of(0, 4), Shared)
    assert not list(SubjectLayer.of(0, 4).parameters())
    assert SubjectLayer.of(2, 4).weight.shape == (2, 4, 4)

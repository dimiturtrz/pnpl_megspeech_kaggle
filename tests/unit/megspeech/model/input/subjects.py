import torch

from megspeech.data.raw.runs import DEEP
from megspeech.model.input.subjects import Reading, Shared, SubjectLayer, Unseen

GENERATOR = torch.Generator().manual_seed(0)


def test_subjects() -> None:
    subject = torch.tensor([1, 5, 32])
    assert torch.equal(Reading.OWN.subjects(subject), subject)
    """An unseen reading is past every layer, however many subjects it holds: a 128-row layer reads it through its
    fallback, not its trained row 40."""
    assert Reading.UNSEEN.subjects(subject).tolist() == [torch.iinfo(torch.int64).max] * 3
    layer = SubjectLayer(128, 4, Unseen.MEAN)
    with torch.no_grad():
        layer.weight[40].mul_(5.0)
    x = torch.randn(3, 4, 6, generator=GENERATOR)
    assert torch.allclose(layer.forward(x, Reading.UNSEEN.subjects(subject)), (1 + 4 / 128) * x)


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


def test_subject_layer_forward_unseen() -> None:
    """A subject past every trained one is read through the layer's `unseen` matrix."""
    layer = SubjectLayer(3, 4, Unseen.MEAN)
    with torch.no_grad():
        layer.weight.mul_(torch.tensor([1.0, 2.0, 3.0])[:, None, None])
    x = torch.randn(2, 4, 6, generator=GENERATOR)
    assert torch.allclose(layer.forward(x, torch.tensor([3, 39])), 2 * x)


def test_shrink() -> None:
    """`keep` 1 changes nothing, 0 collapses every matrix onto the fallback, between interpolates."""
    weight = torch.randn(3, 4, 4, generator=GENERATOR)
    layer = SubjectLayer(3, 4)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.shrink(1.0, Unseen.DEEP)
    assert torch.equal(layer.weight, weight)
    layer.shrink(0.5, Unseen.DEEP)
    assert torch.allclose(layer.weight, 0.5 * weight + 0.5 * weight[DEEP])
    layer.shrink(0.0, Unseen.DEEP)
    assert torch.allclose(layer.weight, weight[DEEP].expand_as(weight))


def test_unseen_matrix() -> None:
    weight = torch.randn(3, 4, 4, generator=GENERATOR)
    assert torch.equal(Unseen.IDENTITY.matrix(weight), torch.eye(4))
    assert torch.equal(Unseen.DEEP.matrix(weight), weight[DEEP])
    assert torch.allclose(Unseen.MEAN.matrix(weight), weight.mean(dim=0))


def test_shared_forward() -> None:
    x = torch.randn(2, 4, 3, generator=GENERATOR)
    assert torch.equal(Shared().forward(x, torch.tensor([0, 5])), x)


def test_targets() -> None:
    weight = torch.randn(3, 2, 2)
    assert torch.equal(Unseen.MEAN.targets(weight), weight.mean(dim=0).expand_as(weight))
    assert torch.equal(Unseen.DEEP.targets(weight), weight[DEEP].expand_as(weight))
    assert torch.equal(Unseen.SOURCE.targets(weight)[DEEP], weight[DEEP])
    assert torch.allclose(Unseen.SOURCE.targets(weight)[1], weight[1:].mean(dim=0), atol=1e-6)


def test_dropped() -> None:
    """Training reads every row through the mean at chance 1 and none at 0; evaluation never drops."""
    layer = SubjectLayer(3, 4, drop=1.0)
    subject = torch.tensor([0, 2, 1])
    assert layer.dropped(subject, 4).tolist() == [4, 4, 4]
    with torch.no_grad():
        layer.weight.mul_(torch.tensor([1.0, 2.0, 3.0])[:, None, None])
    x = torch.randn(3, 4, 6, generator=GENERATOR)
    assert torch.allclose(layer.forward(x, subject), 2 * x)
    layer.eval()
    assert torch.equal(layer.dropped(subject, 4), subject)
    assert torch.equal(SubjectLayer(3, 4).dropped(subject, 4), subject)

import pytest
import torch
from torch import nn

from megspeech.data.raw.runs import DEEP
from megspeech.model.input.subjects import SubjectLayer, Unseen
from megspeech.training.regularise.anchor import Anchor

GENERATOR = torch.Generator().manual_seed(0)
STEPS = 200
LR = 0.01


def anchored() -> tuple[SubjectLayer, nn.Linear, nn.Sequential]:
    layer, head = SubjectLayer(3, 4), nn.Linear(4, 4)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(3, 4, 4, generator=GENERATOR))
    return layer, head, nn.Sequential(layer, head)


def test_of() -> None:
    """Pulling an ungraded deviation through the whole schedule leaves `keep` of it."""
    assert Anchor.of(1.0, LR, STEPS).strength == 0
    anchor = Anchor.of(0.5, LR, STEPS)
    layer, _, model = anchored()
    weight = layer.weight.detach().clone()
    probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=LR)
    schedule = torch.optim.lr_scheduler.OneCycleLR(probe, LR, total_steps=STEPS)
    for _ in range(STEPS):
        anchor.step(model, float(schedule.get_last_lr()[0]))
        probe.step()
        schedule.step()
    deviation = layer.weight - layer.weight[DEEP]
    assert torch.allclose(deviation, 0.5 * (weight - weight[DEEP]), rtol=0.01, atol=1e-5)


def test_groups() -> None:
    layer, head, model = anchored()
    rest, subject = Anchor(1.0).groups(model, 0.05)
    assert subject == {"params": [layer.weight], "weight_decay": 0}
    assert rest == {"params": list(head.parameters())}
    assert Anchor().groups(model, 0.05)[1]["weight_decay"] == pytest.approx(0.05)


def test_step() -> None:
    layer, _, model = anchored()
    weight = layer.weight.detach().clone()
    Anchor(10.0).step(model, 0.05)
    assert torch.allclose(layer.weight, 0.5 * weight + 0.5 * weight[DEEP])
    pulled = layer.weight.detach().clone()
    Anchor().step(model, 1.0)
    assert torch.equal(layer.weight, pulled)


def test_step_toward_mean() -> None:
    layer, _, model = anchored()
    weight = layer.weight.detach().clone()
    Anchor(10.0, Unseen.MEAN).step(model, 0.05)
    assert torch.allclose(layer.weight, 0.5 * weight + 0.5 * weight.mean(dim=0))
    assert Anchor.of(0.5, LR, STEPS, Unseen.MEAN).toward is Unseen.MEAN
    assert Anchor.of(0.5, LR, STEPS).toward is Unseen.DEEP


def test_step_toward_source() -> None:
    layer = SubjectLayer(14, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(14, 3, 3, generator=GENERATOR))
    weight = layer.weight.detach().clone()
    Anchor(10.0, Unseen.SOURCE).step(nn.Sequential(layer), 0.05)
    assert torch.equal(layer.weight[DEEP], weight[DEEP])
    assert torch.allclose(layer.weight[1], 0.5 * weight[1] + 0.5 * weight[1:13].mean(dim=0), atol=1e-6)
    assert torch.allclose(layer.weight[13], weight[13], atol=1e-6)

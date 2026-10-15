import torch
from torch import nn

from megspeech.model.input.subjects import SubjectLayer
from megspeech.training.regularise.weights.pull import Pull

GENERATOR = torch.Generator().manual_seed(0)
STEPS = 200
LR = 0.01


def pulled() -> tuple[SubjectLayer, nn.Linear, nn.Sequential]:
    layer, head = SubjectLayer(3, 4), nn.Linear(4, 4)
    return layer, head, nn.Sequential(layer, head)


def drift(head: nn.Linear) -> None:
    with torch.no_grad():
        head.weight.add_(torch.randn(4, 4, generator=GENERATOR))


def test_of() -> None:
    """Pulling an ungraded drift through the whole schedule leaves `keep` of it; the subject matrices are skipped."""
    layer, head, model = pulled()
    params = (layer.weight, head.weight)
    assert Pull.of(1.0, LR, STEPS, model, params).params == ()
    pull = Pull.of(0.5, LR, STEPS, model, params)
    assert pull.params == (head.weight,)
    start = head.weight.detach().clone()
    drift(head)
    moved = head.weight.detach().clone()
    probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=LR)
    schedule = torch.optim.lr_scheduler.OneCycleLR(probe, LR, total_steps=STEPS)
    for _ in range(STEPS):
        pull.step(model, float(schedule.get_last_lr()[0]))
        probe.step()
        schedule.step()
    assert torch.allclose(head.weight - start, 0.5 * (moved - start), rtol=0.01, atol=1e-5)


def test_step() -> None:
    _, head, model = pulled()
    start = head.weight.detach().clone()
    pull = Pull((head.weight,), (start,), 10.0)
    drift(head)
    moved = head.weight.detach().clone()
    pull.step(model, 0.05)
    assert torch.allclose(head.weight, 0.5 * moved + 0.5 * start)
    Pull().step(model, 1.0)
    assert torch.allclose(head.weight, 0.5 * moved + 0.5 * start)

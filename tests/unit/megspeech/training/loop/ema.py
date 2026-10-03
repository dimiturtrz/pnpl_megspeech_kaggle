import torch
from torch import nn

from megspeech.training.loop.ema import Ema


def model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(3, 2)


def test_of() -> None:
    net = model()
    ema = Ema.of(net, 0.9)
    assert ema.decay == 0.9
    assert all(torch.equal(s, p) for s, p in zip(ema.shadow, net.parameters(), strict=True))
    assert all(s.data_ptr() != p.data_ptr() for s, p in zip(ema.shadow, net.parameters(), strict=True))


def test_step() -> None:
    net = model()
    start = [p.detach().clone() for p in net.parameters()]
    averaged, exact = Ema.of(net, 0.75), Ema.of(net, 0.0)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(4.0)
    averaged.step(net, 0.0)
    exact.step(net, 0.0)
    assert all(torch.allclose(s, b + 1.0) for s, b in zip(averaged.shadow, start, strict=True))
    assert all(torch.equal(s, p) for s, p in zip(exact.shadow, net.parameters(), strict=True))


def test_applied() -> None:
    net = model()
    ema = Ema.of(net, 0.5)
    own = [p.detach().clone() for p in net.parameters()]
    pointers = [p.data_ptr() for p in net.parameters()]
    ema.shadow[0].fill_(7.0)
    with ema.applied(net) as inside:
        assert torch.equal(next(inside.parameters()), torch.full((2, 3), 7.0))
    assert all(torch.equal(p, o) for p, o in zip(net.parameters(), own, strict=True))
    assert [p.data_ptr() for p in net.parameters()] == pointers


def test_state() -> None:
    """BatchNorm running statistics are averaged with the weights; its integer batch counter is not."""
    net = nn.BatchNorm1d(2)
    state = Ema.state(net)
    assert len(state) == len(list(net.parameters())) + 2
    assert all(t.is_floating_point() for t in state)
    ema, running_mean = Ema.of(net, 0.5), state[2]
    running_mean.fill_(4.0)
    ema.step(net, 0.0)
    with ema.applied(net):
        assert running_mean.tolist() == [2.0, 2.0]
    assert running_mean.tolist() == [4.0, 4.0]


def test_copy() -> None:
    into = [torch.zeros(2), torch.zeros(3)]
    Ema.copy(into, [torch.ones(2), torch.full((3,), 2.0)])
    assert into[0].tolist() == [1.0, 1.0]
    assert into[1].tolist() == [2.0, 2.0, 2.0]

import torch
from torch import nn

from megspeech.training.loop.ema import Ema, Unaveraged


def model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(3, 2)


def test_hook() -> None:
    """Decay 0 holds no shadow: its average is the model itself."""
    net = model()
    assert type(Ema.hook(net, 0.0)) is Unaveraged
    averaged = Ema.hook(net, 0.9)
    assert isinstance(averaged, Ema)
    assert averaged.decay == 0.9


def test_unaveraged_step() -> None:
    net = model()
    own = [p.detach().clone() for p in net.parameters()]
    Unaveraged().step(net, 1.0)
    assert all(torch.equal(p, o) for p, o in zip(net.parameters(), own, strict=True))


def test_unaveraged_applied() -> None:
    net = model()
    with Unaveraged().applied(net) as inside:
        assert inside is net


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


def test_ema_applied() -> None:
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


def test_live() -> None:
    net = model()
    ema = Ema.of(net, 0.5)
    assert ema.live(net) is ema.live(net)
    assert all(
        a is not b and a.data_ptr() == b.data_ptr() for a, b in zip(ema.live(net), net.parameters(), strict=True)
    )
    assert ema.live(model()) is not ema.live(net)


def test_step_matches_rebuilt_state() -> None:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Linear(6, 2))
    cached = Ema.of(net, 0.9)
    rebuilt = [t.clone() for t in Ema.state(net)]
    for _ in range(20):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p))
        net(torch.randn(8, 4))
        cached.step(net, 0.0)
        torch._foreach_lerp_(rebuilt, Ema.state(net), 0.1)
    assert all(torch.equal(a, b) for a, b in zip(cached.shadow, rebuilt, strict=True))

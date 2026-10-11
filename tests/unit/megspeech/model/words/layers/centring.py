import torch
from torch import nn

from megspeech.model.words.layers.centring import RunCentring

GENERATOR = torch.Generator().manual_seed(0)
WIDTH = 4
MASK = torch.tensor([[True, True, False], [True, True, True]])
RUN = torch.tensor([[0, 1, -1], [1, 0, 1]])


def embedded() -> torch.Tensor:
    return (torch.randn(2, 3, WIDTH, generator=GENERATOR) + 5.0) * MASK[..., None]


def centring(*, active: bool = True) -> RunCentring:
    return RunCentring(WIDTH, active=active, momentum=0.5)


def run_means(x: torch.Tensor) -> list[torch.Tensor]:
    return [x[MASK & RUN.eq(run)].mean(0) for run in (0, 1)]


def test_reserve() -> None:
    module = centring()
    module.reserve(3)
    assert module.mean.shape == (3, WIDTH)
    assert not module.seen.any()
    assert not module.state_dict()


def test_forward() -> None:
    """Inactive is the input itself; active, each run's words have zero mean, padding stays zero, and a run's
    mean is its own words' alone."""
    x = embedded()
    assert centring(active=False).forward(x, MASK, RUN) is x
    out = centring().eval().forward(x, MASK, RUN)
    for mean in run_means(out):
        assert torch.allclose(mean, torch.zeros(WIDTH), atol=1e-5)
    assert out[~MASK].eq(0).all()
    expected = x[0, 0] - x[MASK & RUN.eq(0)].mean(0)
    assert torch.allclose(out[0, 0], expected, atol=1e-5)


def test_forward_gradient() -> None:
    """The mean is held out of the gradient: each input's gradient is the output's, unscaled."""
    x = embedded().requires_grad_()
    module = centring().train()
    module.reserve(2)
    module.forward(x, MASK, RUN).sum().backward()
    grad = x.grad
    assert grad is not None
    assert torch.equal(grad, torch.ones_like(grad))


def test_centre() -> None:
    x = embedded()
    module = centring().eval()
    means = module.centre(x, MASK, RUN.clamp(min=0))
    assert torch.allclose(means, torch.stack(run_means(x)), atol=1e-5)


def test_update() -> None:
    """A run's first batch sets its mean, later ones move it by 1 - momentum, an absent run keeps its own."""
    module = centring()
    module.reserve(3)
    first = module.update(torch.tensor([[2.0] * WIDTH, [0.0] * WIDTH, [0.0] * WIDTH]), torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(first[0], torch.ones(WIDTH))
    assert torch.equal(module.seen, torch.tensor([True, False, False]))
    moved = module.update(torch.tensor([[9.0] * WIDTH, [4.0] * WIDTH, [0.0] * WIDTH]), torch.tensor([3.0, 1.0, 0.0]))
    assert torch.equal(moved[0], torch.full((WIDTH,), 0.5 * 1.0 + 0.5 * 3.0))
    assert torch.equal(moved[1], torch.full((WIDTH,), 4.0))
    assert torch.equal(moved[2], torch.zeros(WIDTH))


def test_training_reads_moving_mean() -> None:
    x = embedded()
    module = centring().train()
    module.reserve(2)
    out = module.forward(x, MASK, RUN)
    assert torch.allclose(module.mean, torch.stack(run_means(x)), atol=1e-5)
    for mean in run_means(out):
        assert torch.allclose(mean, torch.zeros(WIDTH), atol=1e-5)


def test_runs() -> None:
    assert RunCentring.runs(RUN.clamp(min=0)) == 2
    assert RunCentring.runs(torch.zeros(0, dtype=torch.long)) == 0


def test_tally() -> None:
    x = embedded()
    sums, counts = RunCentring.tally(x, MASK, RUN.clamp(min=0), 3)
    assert torch.equal(counts, torch.tensor([2.0, 3.0, 0.0]))
    assert torch.allclose(sums[1], x[MASK & RUN.eq(1)].sum(0))


def test_collect() -> None:
    """Collecting gathers sums and centres nothing."""
    x = embedded()
    module = centring().eval()
    module.collect()
    assert torch.equal(module.forward(x, MASK, RUN), x)
    assert module.gathered is not None
    assert len(module.gathered) == 1


def test_settle() -> None:
    """The settled mean pools every gathered batch, not each batch's own."""
    first, second = embedded(), embedded()
    module = centring().eval()
    module.collect()
    module.forward(first, MASK, RUN)
    module.forward(second[:, :2], MASK[:, :2], torch.zeros(2, 2, dtype=torch.long))
    module.settle()
    pooled = torch.cat([first[MASK & RUN.eq(0)], second[:, :2][MASK[:, :2]]]).mean(0)
    assert module.fixed is not None
    assert torch.allclose(module.fixed[0], pooled, atol=1e-5)
    assert module.gathered is None


def test_release() -> None:
    module = centring().eval()
    module.collect()
    module.settle()
    module.release()
    assert module.fixed is None
    assert module.gathered is None


def test_calibrated() -> None:
    """Two passes, the first gathering, give each word less its run's mean over every batch; inactive never sweeps."""
    batches = [embedded(), embedded()]
    module = centring()
    model = nn.Sequential(module).eval()

    def sweep() -> list[torch.Tensor]:
        return [module.forward(x, MASK, RUN) for x in batches]

    with RunCentring.calibrated(model, sweep):
        out = sweep()
    pooled = torch.cat([x[MASK & RUN.eq(0)] for x in batches]).mean(0)
    assert torch.allclose(out[1][0, 0], batches[1][0, 0] - pooled, atol=1e-5)
    assert module.fixed is None
    swept: list[int] = []
    with RunCentring.calibrated(nn.Sequential(centring(active=False)), lambda: swept.append(1)):
        pass
    assert not swept

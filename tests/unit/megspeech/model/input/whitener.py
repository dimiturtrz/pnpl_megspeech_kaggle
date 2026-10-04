import torch

from megspeech.model.input.whitener import Whitener

GENERATOR = torch.Generator().manual_seed(0)
MIXING = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.4, 0.0], [0.0, 0.0, 3.0]])


def correlated(n: int) -> torch.Tensor:
    return torch.einsum("dc,nct->ndt", MIXING, torch.randn(n, 3, 20, generator=GENERATOR))


def test_fit() -> None:
    """Fit on many windows, the whitened sensors are uncorrelated at unit variance; inactive, it passes them through."""
    x = correlated(4000)
    whitener = Whitener(3)
    whitener.fit(x)
    flat = whitener(x).transpose(0, 1).flatten(1)
    assert torch.allclose(torch.cov(flat), torch.eye(3), atol=0.05)
    assert torch.allclose(whitener.matrix, whitener.matrix.T, atol=1e-5)
    unwhitened = Whitener(3, active=False)
    unwhitened.fit(x)
    assert torch.equal(unwhitened(x), x)


def test_shrunk() -> None:
    """Many windows shrink barely, toward the population covariance; three windows, rank two at each time point,
    still shrink to an invertible one."""
    population = MIXING @ MIXING.T
    assert torch.allclose(Whitener.shrunk(correlated(4000)).float(), population, atol=0.15)
    few = correlated(3)[:, :, :1]
    assert torch.linalg.eigvalsh(Whitener.shrunk(few)).min() > 0


def test_forward() -> None:
    """Unfit, the whitener is the identity; fit, it mixes sensors and keeps the window's shape."""
    x = correlated(8)
    whitener = Whitener(3)
    assert torch.equal(whitener.forward(x), x)
    whitener.fit(correlated(500))
    assert whitener.forward(x).shape == x.shape
    assert not torch.allclose(whitener.forward(x), x)

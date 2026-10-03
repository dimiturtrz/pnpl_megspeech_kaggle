import dataclasses

import numpy as np
import torch

from megspeech.data.cache.alignment import GATE, Alignment

GENERATOR = torch.Generator().manual_seed(0)
CHANNELS = 4
SAMPLES = 40_000


def mixed(mixing: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
    return mixing @ torch.randn(CHANNELS, SAMPLES, generator=GENERATOR) + offset


def reference() -> tuple[Alignment, torch.Tensor]:
    mixing = torch.eye(CHANNELS) + 0.5 * torch.randn(CHANNELS, CHANNELS, generator=GENERATOR)
    runs = [mixed(mixing, 3.0).numpy() for _ in range(2)]
    return Alignment.of(runs, "cpu"), mixing


def test_of() -> None:
    alignment, mixing = reference()
    assert alignment.median.shape == alignment.spread.shape == (CHANNELS, 1)
    assert torch.allclose(alignment.median, torch.full((CHANNELS, 1), 3.0), atol=0.1)
    covariance = (alignment.root @ alignment.root).float()
    expected = mixing @ mixing.T / (alignment.spread * alignment.spread.T)
    assert torch.allclose(covariance, expected, rtol=0.1, atol=0.05)


def test_robust() -> None:
    x = torch.arange(101, dtype=torch.float32).expand(2, -1)
    median, spread = Alignment.robust(x)
    assert torch.equal(median, torch.full((2, 1), 50.0))
    assert torch.equal(spread, torch.full((2, 1), 50.0))
    assert (Alignment.robust(torch.zeros(1, 10))[1] > 0).all()


def test_covariance() -> None:
    z = torch.randn(CHANNELS, 1000, generator=GENERATOR)
    spiked = torch.cat([z, torch.full((CHANNELS, 1), 2 * GATE)], dim=1)
    assert torch.allclose(Alignment.covariance(spiked), Alignment.covariance(z))
    assert Alignment.covariance(z).dtype == torch.float64


def test_power() -> None:
    basis = torch.linalg.qr(torch.randn(CHANNELS, CHANNELS, generator=GENERATOR, dtype=torch.float64))[0]
    full = basis @ torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)) @ basis.T
    root = Alignment.power(full, 0.5)
    assert torch.allclose(root @ root, full)
    deficient = basis[:, :2] @ basis[:, :2].T
    inverse = Alignment.power(deficient, -0.5)
    assert torch.allclose(inverse, deficient, atol=1e-9)


def test_applies() -> None:
    names = ("sub-0_ses-1_task-Sherlock1_run-1", "subj00_holdout01_sentence", "sub-10_ses-1_task-Sherlock1_run-1")
    alignment, _ = reference()
    assert [alignment.applies(name) for name in (*names, "subj01_holdout2_word")] == [False, False, True, True]
    assert all(dataclasses.replace(alignment, exempt=()).applies(name) for name in names)


def test_align() -> None:
    alignment, _ = reference()
    other = torch.diag(torch.tensor([3.0, 0.5, 1.0, 2.0])) + 0.3 * torch.randn(CHANNELS, CHANNELS, generator=GENERATOR)
    run = mixed(other, -7.0)
    alignment.align(run)
    median, _ = Alignment.robust(run)
    assert torch.allclose(median, alignment.median, atol=0.1)
    covariance = Alignment.covariance((run - median) / alignment.spread).float()
    assert torch.allclose(covariance, (alignment.root @ alignment.root).float(), rtol=0.1, atol=0.05)


def test_align_keeps_dtype() -> None:
    alignment, _ = reference()
    run = mixed(torch.eye(CHANNELS)).half()
    alignment.align(run)
    assert run.dtype == torch.float16
    assert np.isfinite(run.float().numpy()).all()

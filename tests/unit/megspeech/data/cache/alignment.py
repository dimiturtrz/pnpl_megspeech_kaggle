import dataclasses
from pathlib import Path

import mne
import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import AS_RECORDED, GATE, Alignment, Geometry
from megspeech.data.raw.fetch import NEUROMAG

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


def deficient() -> tuple[Alignment, torch.Tensor]:
    """A reference whose four channels mix two sources: rank 2, as a Maxwell filter leaves 75 of 306."""
    mixing = torch.randn(CHANNELS, 2, generator=GENERATOR)
    runs = [(mixing @ torch.randn(2, SAMPLES, generator=GENERATOR) + 3.0).numpy() for _ in range(2)]
    return Alignment.of(runs, "cpu"), mixing


def test_projection() -> None:
    alignment, _ = deficient()
    projection = alignment.projection()
    assert projection.basis.shape == (2, CHANNELS)
    assert projection.basis.dtype == torch.float32
    assert torch.allclose(projection.basis @ projection.basis.T, torch.eye(2), atol=1e-5)
    variance = (projection.basis @ alignment.root.float() @ alignment.root.float() @ projection.basis.T).diagonal()
    assert variance[0] >= variance[1]
    assert torch.equal(projection.median, alignment.median)
    assert torch.equal(projection.spread, alignment.spread)


def test_rank() -> None:
    assert deficient()[0].projection().rank == 2
    alignment = reference()[0]
    full, leading = alignment.projection(), alignment.projection(1)
    assert full.rank == CHANNELS
    assert leading.rank == 1
    assert torch.allclose(leading.basis, full.basis[:1], atol=1e-5)


def test_selection(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    NEUROMAG.home(root).parent.mkdir(parents=True)
    info = mne.create_info(["MEG 0111", "MEG 0121", "MEG 0131", "MEG 2041"], 100.0, "mag")
    mne.io.RawArray(np.zeros((CHANNELS, 10)), info, verbose=False).save(NEUROMAG.home(root), verbose=False)
    alignment = reference()[0]
    selection = alignment.selection(("Left-temporal",), root)
    assert torch.equal(selection.basis, torch.eye(CHANNELS)[[0, 2]])
    assert torch.equal(selection.median, alignment.median)
    assert alignment.selection((), root).rank == 0


def test_projection_project() -> None:
    alignment, mixing = deficient()
    projection = alignment.projection()
    run = mixing @ torch.randn(2, 500, generator=GENERATOR) + alignment.median
    coordinates = projection.project(run)
    assert coordinates.shape == (2, 500)
    restored = projection.basis.T @ coordinates * projection.spread + projection.median
    assert torch.allclose(restored, run, atol=1e-4)
    assert projection.project(run.half()).dtype == torch.float32


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


def test_alignment_align() -> None:
    alignment, _ = reference()
    other = torch.diag(torch.tensor([3.0, 0.5, 1.0, 2.0])) + 0.3 * torch.randn(CHANNELS, CHANNELS, generator=GENERATOR)
    run = mixed(other, -7.0)
    alignment.align(run)
    median, _ = Alignment.robust(run)
    assert torch.allclose(median, alignment.median, atol=0.1)
    covariance = Alignment.covariance((run - median) / alignment.spread).float()
    assert torch.allclose(covariance, (alignment.root @ alignment.root).float(), rtol=0.1, atol=0.05)


def test_rows() -> None:
    alignment, _ = deficient()
    assert AS_RECORDED.rows(CHANNELS) == CHANNELS
    assert Geometry(alignment).rows(CHANNELS) == CHANNELS
    assert Geometry(projection=alignment.projection()).rows(CHANNELS) == 2


def test_geometry_align() -> None:
    alignment, _ = reference()
    other = "sub-10_ses-1_task-Sherlock1_run-1"
    run = mixed(torch.eye(CHANNELS), -7.0)
    unchanged = run.clone()
    AS_RECORDED.align(run, other)
    Geometry(alignment).align(run, "sub-0_ses-1_task-Sherlock1_run-1")
    assert torch.equal(run, unchanged)
    Geometry(alignment).align(run, other)
    expected = unchanged.clone()
    alignment.align(expected)
    assert torch.equal(run, expected)


def test_geometry_project() -> None:
    alignment, mixing = deficient()
    run = mixing @ torch.randn(2, 100, generator=GENERATOR) + alignment.median
    assert AS_RECORDED.project(run) is run
    projected = Geometry(projection=alignment.projection()).project(run)
    assert torch.equal(projected, alignment.projection().project(run))


def test_align_keeps_dtype() -> None:
    alignment, _ = reference()
    run = mixed(torch.eye(CHANNELS)).half()
    alignment.align(run)
    assert run.dtype == torch.float16
    assert np.isfinite(run.float().numpy()).all()

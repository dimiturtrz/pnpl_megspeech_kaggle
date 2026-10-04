from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from megspeech.data.batching.windows import Windows
from megspeech.evaluation.topography import Evoked, Helmet

SIGNAL = torch.tensor([[1.0, 1.0], [0.0, 3.0]])


def evoked() -> Evoked:
    """Four words: sensor 0 always 1 then 1, sensor 1 swinging 0 +- 1 then 3 +- 1."""
    noise = torch.tensor([1.0, -1.0, 1.0, -1.0])[:, None, None] * torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    return Evoked.of(Windows.stacked(SIGNAL + noise, centred=False), chunk=3)


def test_of() -> None:
    result = evoked()
    assert torch.allclose(result.mean, SIGNAL)
    assert torch.allclose(result.spread, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert result.count == 4


def test_t() -> None:
    t = evoked().t()
    assert torch.allclose(t[1], torch.tensor([0.0, 6.0]))
    assert t[0].gt(1e6).all()


def test_peaks() -> None:
    assert evoked().peaks(0, 1)[1] == pytest.approx(0.0)
    assert evoked().peaks(0, 2)[1] == pytest.approx(6.0)


def test_read(tmp_path: Path) -> None:
    info = mne.create_info(["MEG 0112", "MEG 0111", "STI 014"], 100.0, ["grad", "mag", "stim"])
    path = tmp_path / "helmet_raw.fif"
    mne.io.RawArray(np.zeros((3, 10)), info, verbose=False).save(path, verbose=False)
    helmet = Helmet.read(path)
    assert helmet.names == ("MEG0111", "MEG0112")
    assert helmet.positions.shape == (2, 3)


def test_within() -> None:
    helmet = Helmet(("MEG0111", "MEG2041"), np.zeros((2, 3)))
    assert helmet.within(("Left-temporal",)).tolist() == [True, False]
    assert helmet.within(()).tolist() == [False, False]


def test_compact() -> None:
    assert Helmet.compact("MEG 0111") == "MEG0111"

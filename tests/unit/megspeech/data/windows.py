from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import resample_poly

from megspeech.data import continuous
from megspeech.data.continuous import RATE, Continuous, Decimator
from megspeech.data.corpus import Corpus
from megspeech.data.epochs import DECIMATE, PICO, SFREQ, WINDOW
from megspeech.data.grouping import Grouping
from megspeech.data.runs import Run
from megspeech.data.vocabulary import PRIMARY
from megspeech.data.windows import Windows

CHANNELS = 4
SECONDS = 6
EDGE = 10
"""Output samples `resample_poly`'s default filter reaches past a window's edge: 10 * DECIMATE taps / DECIMATE."""
HALF_ULP = 2.0**-11
"""float16's relative rounding error."""


def signal(length: int = 8) -> Windows:
    meg = torch.arange(2 * length, dtype=torch.float32).reshape(2, length)
    return Windows(meg, torch.tensor([0, 2, 4]), length=3, offset=1)


def test_stacked() -> None:
    x = torch.arange(2 * CHANNELS * 3, dtype=torch.float32).reshape(2, CHANNELS, 3)
    windows = Windows.stacked(x)
    assert len(windows) == 2
    assert torch.equal(windows(torch.tensor([1, 0])), x[[1, 0]] - x[[1, 0]].mean(dim=-1, keepdim=True))


def test_device() -> None:
    assert signal().device == torch.device("cpu")


def test_call() -> None:
    windows = signal()
    cut = windows(torch.tensor([2, 0]))
    assert cut.shape == (2, 2, 3)
    assert cut[0, 1].tolist() == [-1.0, 0.0, 1.0]
    assert cut[1].tolist() == [[-1.0, 0.0, 1.0]] * 2


def recording(root: Path) -> tuple[Run, list[tuple[str, float]]]:
    """A run with heard words on and off the 50 Hz grid, out of the vocabulary, and too close to either end."""
    run = Run(root / "derivatives" / "serialised" / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5")
    run.meg.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    drift = np.linspace(0, 3, SECONDS * SFREQ)
    with h5py.File(run.meg, "w") as stored:
        stored["data"] = ((rng.normal(size=(CHANNELS, SECONDS * SFREQ)) + drift) * 1e-12).astype(np.float32)
        stored.attrs["sample_frequency"] = SFREQ
    heard = [
        ("the", -0.1),
        ("the", 0.5),
        ("zebra", 1.0),
        ("a", 1.52),
        ("is", 2.04),
        ("None", 2.5),
        ("it", 3.0),
        ("i", 4.013),
    ]
    heard += [("to", SECONDS - 1.0), ("was", SECONDS - 0.5)]
    rows = [("word", word, 0, onset) for word, onset in heard]
    run.events.parent.mkdir(parents=True)
    columns = ["kind", "segment", "sentenceidx", "timemeg"]
    pd.DataFrame(rows, columns=columns).to_csv(run.events, sep="\t", index=False)
    return run, heard


def test_call_reproduces_the_epoch_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gathered window equals a 1 s epoch of raw MEG from `floor(onset * SFREQ)` decimated on its own.

    Equal event set, labels and order; equal values on samples both decimations compute from the same inputs —
    away from `resample_poly`'s zero-padded window edges, on onsets on the 50 Hz grid — up to float16 rounding.
    The cache rounds each sample by at most `HALF_ULP * level` (`level` the window's largest cached magnitude), the
    output by at most `HALF_ULP * 2 * level`; centring both paths on the same samples at most doubles that: 6.
    """
    monkeypatch.setattr(continuous, "ProcessPoolExecutor", Pool)
    run, heard = recording(tmp_path)
    Decimator(tmp_path / "cache").build([run])
    source = Continuous.load(tmp_path / "cache", "cpu")
    words = Corpus.words(source.words, Grouping.GAP)
    gathered = Corpus.split(source, words, Grouping.GAP, np.ones(len(words))).windows(torch.arange(len(words)))

    with h5py.File(run.meg) as stored:
        raw = stored["data"][:]
    onset = np.array([at for _, at in heard])
    start = (onset * SFREQ).astype(np.int64)
    labels = PRIMARY.labels([word for word, _ in heard])
    old = (labels >= 0) & (start >= 0) & (start + WINDOW <= raw.shape[1])
    cut = raw[:, start[old, None] + np.arange(WINDOW)].transpose(1, 0, 2).astype(np.float64) * PICO
    epoch = resample_poly(cut, 1, DECIMATE, axis=-1)

    assert words["onset"].tolist() == onset[old].tolist()
    assert words["target"].tolist() == labels[old].tolist()
    aligned = start[old] % DECIMATE == 0
    assert aligned.sum() >= 3
    inner = slice(EDGE, RATE - EDGE)
    new = gathered.float().numpy()[aligned][..., inner]
    reference = epoch[aligned][..., inner]
    cached = source.meg.float().numpy()[
        :, source.words["at"].to_numpy()[words.index[aligned]][:, None] + np.arange(RATE)
    ]
    level = np.abs(cached).max(axis=(0, 2))[:, None, None]
    tolerance = 6 * HALF_ULP * level
    centred = (new - new.mean(axis=-1, keepdims=True)) - (reference - reference.mean(axis=-1, keepdims=True))
    assert (np.abs(centred) <= tolerance).all()


class Pool:
    def __init__(self, _: int) -> None:
        pass

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def map(self, function: Callable[[Run], Path], items: list[Run]) -> list[Path]:
        return [function(item) for item in items]

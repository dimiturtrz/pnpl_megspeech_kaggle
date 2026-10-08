from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import resample_poly

from megspeech.data.batching.windows import Windows
from megspeech.data.cache import run_cache
from megspeech.data.cache.run_cache import DeviceRecording, RunCache
from megspeech.data.config import DataConfig
from megspeech.data.constants import DECIMATE, PICO, RATE, SFREQ, WINDOW
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import Grouping

CHANNELS = 4
SECONDS = 6
EDGE = 10
"""Output samples `resample_poly`'s default filter reaches past a window's edge: 10 * DECIMATE taps / DECIMATE."""
HALF_ULP = 2.0**-11
"""float16's relative rounding error."""


def signal(length: int = 8) -> Windows:
    meg = torch.arange(2 * length, dtype=torch.float32).reshape(2, length)
    at = torch.tensor([0, 2, 4])
    return Windows(meg, at, at + 4, length=3, offset=1)


def test_stacked() -> None:
    x = torch.arange(2 * CHANNELS * 3, dtype=torch.float32).reshape(2, CHANNELS, 3)
    windows = Windows.stacked(x)
    assert len(windows) == 2
    assert torch.equal(windows(torch.tensor([1, 0])), x[[1, 0]] - x[[1, 0]].mean(dim=-1, keepdim=True))
    features = x[:, :, :1]
    assert torch.equal(Windows.stacked(features, centred=False)(torch.tensor([1, 0])), features[[1, 0]])


def test_device() -> None:
    assert signal().device == torch.device("cpu")


def test_call() -> None:
    windows = signal()
    cut = windows(torch.tensor([2, 0]))
    assert cut.shape == (2, 2, 3)
    assert cut[0, 1].tolist() == [-1.0, 0.0, 1.0]
    assert cut[1].tolist() == [[-1.0, 0.0, 1.0]] * 2
    bounded = Windows(windows.meg, windows.at, torch.tensor([3, 8, 8]), length=3, offset=1)
    assert bounded(torch.tensor([0]))[0, 0].tolist() == [-0.5, 0.5, 0.0]


def test_call_lead_reads_nothing_before_begin() -> None:
    """A window led before its onset zeroes the samples before its `begin`; without one, before the signal."""
    meg = torch.arange(20.0)[None]
    at = torch.tensor([0, 6])
    led = Windows(meg, at, at + 4, length=6, offset=-2, centred=False, begin=torch.tensor([0, 5]))
    assert led(torch.tensor([0, 1]))[:, 0].tolist() == [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
    bare = Windows(meg, at, at + 4, length=6, offset=-2, centred=False)
    assert bare(torch.tensor([0, 1]))[:, 0].tolist() == [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]


def test_shifted() -> None:
    windows = signal()
    moved = windows.shifted(1)
    assert moved.at.tolist() == [1, 3, 5]
    assert torch.equal(moved.end, windows.end)
    assert torch.equal(
        moved(torch.tensor([0])), Windows(windows.meg, windows.at + 1, windows.end, 3, 1)(torch.tensor([0]))
    )


def test_first() -> None:
    at = torch.tensor([3, 7])
    assert Windows(torch.zeros(1, 9), at, at).first(torch.tensor([1])).tolist() == [[0]]
    assert Windows(torch.zeros(1, 9), at, at, begin=torch.tensor([1, 5])).first(torch.tensor([1, 0])).tolist() == [
        [5],
        [1],
    ]


def test_call_bounded_past_the_window_is_the_plain_centred_cut() -> None:
    """Every bound at or past its window's end: bit for bit the cut centred by its plain mean, in float16 too."""
    meg = torch.randn(5, 400, generator=torch.Generator().manual_seed(0)).half()
    at = torch.tensor([0, 37, 150, 349])
    words = torch.tensor([3, 0, 2, 1])
    cut = meg[:, at[words, None] + torch.arange(RATE)].transpose(0, 1).float()
    plain = (cut - cut.mean(dim=-1, keepdim=True)).half()
    assert torch.equal(Windows(meg, at, at + RATE)(words), plain)
    assert torch.equal(Windows(meg, at, torch.full_like(at, 400))(words), plain)


def test_call_jump_past_float16_range_saturates() -> None:
    meg = torch.tensor([[-60000.0] * 7 + [60000.0]], dtype=torch.float16)
    at = torch.tensor([0])
    cut = Windows(meg, at, at + 8, 8)(at)
    assert cut[0, 0].tolist() == [-15000.0] * 7 + [torch.finfo(torch.float16).max]


def test_call_baseline_centres_on_the_window_head() -> None:
    """A `baseline` centres on the valid samples among the window's first `baseline`; a bound inside it narrows it."""
    meg = torch.arange(20.0)[None].repeat(2, 1)
    at = torch.tensor([0, 5])
    both = torch.tensor([0, 1])
    full = Windows(meg, at, at + 10, 10, baseline=4)(both)
    assert full[0, 0].tolist() == [x - 1.5 for x in range(10)]
    assert full[1, 1].tolist() == [x - 6.5 for x in range(5, 15)]
    short = Windows(meg, at, at + 2, 10, baseline=4)(torch.tensor([0]))
    assert short[0, 0].tolist() == [-0.5, 0.5] + [0.0] * 8
    assert torch.equal(Windows(meg, at, at + 10, 10, baseline=10)(both), Windows(meg, at, at + 10, 10)(both))


def test_call_jitter_moves_each_start_within_its_bound() -> None:
    """A jittered window is the plain window of a start moved by at most `jitter`; draws differ across words."""
    meg = torch.randn(3, 200, generator=torch.Generator().manual_seed(0))
    at = torch.full((64,), 50)
    end = at + RATE
    words = torch.arange(64)
    torch.manual_seed(0)
    cut = Windows(meg, at, end, length=10)(words, jitter=2)
    shifted = torch.stack([Windows(meg, at + s, end, length=10)(words) for s in range(-2, 3)])
    match = (shifted == cut).flatten(2).all(dim=-1)
    assert torch.all(match.any(dim=0))
    assert len(set(match.int().argmax(dim=0).tolist())) == len(range(-2, 3))


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
    rows = [("word", word, 0, onset, 0.2) for word, onset in heard]
    run.events.parent.mkdir(parents=True)
    columns = ["kind", "segment", "sentenceidx", "timemeg", "duration"]
    pd.DataFrame(rows, columns=columns).to_csv(run.events, sep="\t", index=False)
    return run, heard


def test_call_reproduces_the_epoch_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gathered window equals a 1 s epoch of raw MEG from `floor(onset * SFREQ)` decimated on its own.

    Equal event set, labels and order; equal values on samples both decimations compute from the same inputs —
    away from `resample_poly`'s zero-padded window edges, on onsets on the 50 Hz grid — up to float16 rounding.
    The cache rounds each sample by at most `HALF_ULP * level` (`level` the window's largest cached magnitude), the
    output by at most `HALF_ULP * 2 * level`; centring both paths on the same samples at most doubles that: 6.
    """
    monkeypatch.setattr(run_cache, "ProcessPoolExecutor", Pool)
    run, heard = recording(tmp_path)
    cache = RunCache(tmp_path / "cache")
    cache.build([run])
    source = DeviceRecording.load(cache, "cpu")
    words = RunCache.carried(source.words, Grouping.GAP)
    gathered = Corpus.split(source, words, DataConfig(grouping=Grouping.GAP), np.ones(len(words))).windows(
        torch.arange(len(words))
    )

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
    def __init__(self, _: int, **__: object) -> None:
        pass

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def map(self, function: Callable[[Run], Path], items: list[Run]) -> list[Path]:
        return [function(item) for item in items]


def test_unread() -> None:
    at = torch.tensor([0, 4])
    end = torch.tensor([3, 9])
    meg = torch.zeros(1, 9)
    assert Windows(meg, at, end, 2, offset=2).unread().tolist() == []
    assert Windows(meg, at, end, 2, offset=3).unread().tolist() == [0]


def test_starts() -> None:
    meg, at = torch.zeros(2, 20), torch.tensor([0, 5])
    unbounded = Windows(meg, at, at + 5)
    assert unbounded.starts.tolist() == [0, 0]
    assert unbounded.starts is unbounded.starts
    assert Windows(meg, at, at + 5, begin=torch.tensor([1, 2])).starts.tolist() == [1, 2]

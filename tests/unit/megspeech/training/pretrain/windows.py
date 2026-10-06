from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.stratify import Picked, Stratify
from megspeech.training.pretrain.windows import PartitionError, Pool, Sessions

LENGTH = 50


@pytest.fixture(autouse=True)
def first_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Stratify.split` reads every session's events; its own tests cover it, here the first sessions stand in."""

    def stand_in(source: tuple[Run, ...], count: int) -> Picked:
        return list(source[:count]), list(source[count : 2 * count])

    monkeypatch.setattr(Stratify, "split", staticmethod(stand_in))


def run(subject: int, task: str, session: int, number: int = 1) -> Run:
    return Run(Path(f"sub-{subject}_ses-{session}_task-{task}_run-{number}_meg.h5"))


def runs() -> list[Run]:
    train = [run(0, "Sherlock1", session) for session in range(1, 11)] + [run(0, "MOCHATIMIT", s) for s in (1, 4, 5)]
    held = [run(0, "Sherlock1", 11, 2), run(0, "Sherlock1", 12, 2), run(0, "MOCHATIMIT", 2), run(0, "MOCHATIMIT", 3)]
    return [*train, *held, run(1, "Sherlock1", 11), run(1, "Sherlock1", 1), run(2, "Sherlock1", 12)]


def pool() -> Pool:
    meg = torch.randn(3, 300).half()
    return Pool(DeviceRecording(meg, np.array([0, 100, 300]), ("a", "b"), pd.DataFrame()), LENGTH)


def test_of() -> None:
    data = DataConfig()
    chosen = Sessions.of(data, runs())
    carved = {r.name for r in data.carved(runs())}
    assert chosen.runs
    assert carved
    assert all(r.subject == 0 for r in chosen.runs)
    assert not carved & set(chosen.names)
    assert all(r.partition is Partition.TRAIN for r in chosen.runs)


def test_assert_clean() -> None:
    data = DataConfig()
    everything = runs()
    good = Sessions.of(data, everything).runs
    for bad in (
        run(0, "Sherlock1", 11, 2),
        run(0, "Sherlock1", 12, 2),
        run(1, "Sherlock1", 1),
        run(2, "Sherlock1", 12),
    ):
        with pytest.raises(PartitionError):
            Sessions((*good, bad)).assert_clean(data, everything)
    with pytest.raises(PartitionError):
        Sessions(tuple(data.carved(everything))).assert_clean(data, everything)
    with pytest.raises(PartitionError):
        Sessions(()).assert_clean(data, everything)


def test_limited() -> None:
    chosen = Sessions.of(DataConfig(), runs())
    assert chosen.limited(0).names == sorted(chosen.names)
    assert chosen.limited(2).names == sorted(chosen.names)[:2]


def test_names() -> None:
    assert Sessions((run(0, "Sherlock1", 1),)).names == ["sub-0_ses-1_task-Sherlock1_run-1"]


def test_sizes() -> None:
    assert pool().sizes.tolist() == [100, 200]


def test_total() -> None:
    assert pool().total == 300


def test_onsets() -> None:
    onsets = pool().onsets(2000, torch.Generator().manual_seed(0))
    first, second = onsets < 100, onsets >= 100
    assert (onsets[first] <= 100 - LENGTH).all()
    assert (onsets[second] <= 300 - LENGTH).all()
    assert first.any()
    assert second.any()
    assert abs(second.float().mean().item() - 151 / 202) < 0.05


def test_windows() -> None:
    assert pool().windows(torch.tensor([0, 100, 250])).shape == (3, 3, LENGTH)


def test_sample() -> None:
    assert pool().sample(7, torch.Generator().manual_seed(0)).shape == (7, 3, LENGTH)


def test_short_session() -> None:
    meg = torch.randn(3, 60).half()
    with pytest.raises(ValueError, match="shorter"):
        Pool(DeviceRecording(meg, np.array([0, 10, 60]), ("a", "b"), pd.DataFrame()), LENGTH)

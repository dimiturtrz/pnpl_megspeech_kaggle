"""The MEG pretraining reads: subject 0's continuous training sessions, windows cut at random onsets anywhere."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.windows import Windows
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig, Refit, Val
from megspeech.data.raw.runs import DEEP, Partition, Run


class PartitionError(ValueError):
    """A session pretraining may not read was chosen."""


@dataclass(frozen=True)
class Sessions:
    """The sessions pretraining reads, which no val, local test or Kaggle test MEG may enter."""

    runs: tuple[Run, ...]

    @classmethod
    def of(cls, data: DataConfig, runs: Sequence[Run]) -> "Sessions":
        """Subject 0's `TRAIN`-partition sessions less the carved val: the narrowest training set any recipe uses, so
        no session a carved or an official validation selects on, nor any a local test reports on, is ever read."""
        carved = cls.carving(data)
        chosen = cls(tuple(carved.trained(runs)))
        chosen.assert_clean(carved, runs)
        return chosen

    @staticmethod
    def carving(data: DataConfig) -> DataConfig:
        """`data` as the narrowest recipe reads it: subject 0, the carved val, no refit, nothing held back else."""
        return replace(data, val=Val.CARVED, refit=Refit.NONE, subjects=(DEEP,), train_share=1.0, tasks=())

    def assert_clean(self, data: DataConfig, runs: Sequence[Run]) -> None:
        """Raise `PartitionError` unless every session is subject 0's, `TRAIN`, and neither carved val nor official
        validation or test."""
        withheld = {r.name for r in data.carved(runs)} | {
            r.name for r in runs if r.partition in (Partition.VALIDATION, Partition.TEST, Partition.POOL)
        }
        foreign = [r.name for r in self.runs if r.subject != DEEP or r.partition is not Partition.TRAIN]
        leaked = sorted(withheld & {r.name for r in self.runs})
        if foreign or leaked or not self.runs:
            raise PartitionError(f"pretraining sessions: {len(self.runs)}, foreign {foreign}, withheld {leaked}")

    def limited(self, count: int) -> "Sessions":
        """The first `count` sessions in name order; every one when `count` is 0."""
        ordered = tuple(sorted(self.runs, key=lambda r: r.name))
        return Sessions(ordered[:count] if count else ordered)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.runs]


@dataclass(frozen=True)
class Pool:
    """Sessions end to end in one tensor, each window drawn wholly inside one session at a uniformly random onset."""

    recording: DeviceRecording
    length: int

    def __post_init__(self) -> None:
        if (self.sizes < self.length).any():
            raise ValueError(f"a session is shorter than the {self.length}-sample window")

    @property
    def sizes(self) -> Int[Tensor, " r"]:
        offsets = torch.from_numpy(self.recording.offsets)
        return offsets[1:] - offsets[:-1]

    @property
    def total(self) -> int:
        """Samples across every session."""
        return int(self.recording.offsets[-1])

    def onsets(self, count: int, generator: torch.Generator) -> Int[Tensor, " k"]:
        """`count` window starts, each session weighted by the onsets it holds."""
        room = self.sizes - self.length + 1
        ends = room.cumsum(0)
        draw = torch.randint(int(ends[-1]), (count,), generator=generator)
        session = torch.searchsorted(ends, draw, right=True)
        starts = torch.from_numpy(self.recording.offsets)[:-1]
        return starts[session] + draw - (ends[session] - room[session])

    def windows(self, onsets: Int[Tensor, " k"]) -> Float[Tensor, "k c t"]:
        """The windows at `onsets`, centred as every training window is."""
        cut = Windows(self.recording.meg, onsets, onsets + self.length, self.length)
        return cut(torch.arange(len(onsets), device=onsets.device))

    def sample(self, count: int, generator: torch.Generator) -> Float[Tensor, "k c t"]:
        return self.windows(self.onsets(count, generator))

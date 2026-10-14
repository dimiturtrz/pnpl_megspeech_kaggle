"""LibriSpeech word onsets as span-wide timing training data: `spans.parquet` cut like a Kaggle test sentence."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import ABSENT, IGNORE, PRIMARY
from megspeech.data.spans.spans import Spans

FILE = "spans.parquet"
SPAN_GAP = 1e4
"""Seconds between consecutive spans on the shared clock `Spans.by_sentence` cuts, far above any span's length."""


class Timed(Protocol):
    """What the timing expert trains on: a label per word, the words of each span, and their onsets."""

    @property
    def y(self) -> Int[Tensor, " n"]: ...

    @property
    def index(self) -> Int[Tensor, "g w"]: ...

    @property
    def onset(self) -> Float[Tensor, "g w"]: ...


@dataclass(frozen=True)
class TimingCorpus:
    """Words labelled in the vocabulary (`IGNORE` outside it), in spans of at most `MAX_WORDS`."""

    y: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]
    onset: Float[Tensor, "g w"]

    @classmethod
    def read(cls, root: DataRoot, name: str) -> "TimingCorpus":
        """The corpus `name` names under processed/, `<name>/spans.parquet`."""
        return cls.from_frame(pd.read_parquet(root.processed(name) / FILE))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "TimingCorpus":
        """A corpus of the rows of `frame` (span, word, onset), spans over `MAX_WORDS` cut into near-equal pieces."""
        ordered = frame.sort_values("span", kind="stable")
        group = pd.factorize(ordered["span"])[0].astype(np.int64)
        clock = ordered["onset"].to_numpy(np.float64) + group * SPAN_GAP
        labels = PRIMARY.labels(ordered["word"])
        spans = Spans.by_sentence(group, clock)
        return cls(
            torch.from_numpy(np.where(labels == ABSENT, IGNORE, labels)),
            torch.from_numpy(spans.index),
            torch.from_numpy(spans.onset).float(),
        )

    def to(self, device: torch.device) -> "TimingCorpus":
        """This corpus on `device`."""
        return TimingCorpus(self.y.to(device), self.index.to(device), self.onset.to(device))

    def describe(self) -> str:
        """Spans, words and the share of words with a vocabulary label."""
        return f"{len(self.index)} spans, {len(self.y)} words, {(self.y != IGNORE).float().mean().item():.4f} labelled"

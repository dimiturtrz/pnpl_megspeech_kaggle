"""Word windows on the device, grouped into spans, and the fixed-shape batches training and scoring draw from them."""

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.epochs import Epocher, Epochs, Window
from megspeech.data.runs import Partition, Run
from megspeech.data.spans import PAD, Spans
from megspeech.data.vocabulary import IGNORE, WORDS

EVAL_SPANS = 256

DENSE = 4
"""Validation spans this long or longer look like holdout sentences (94% of which carry 4-9 words)."""


RECITED = 3
"""Spans this long recur only when a sentence is replayed; 2-word pairs of 50 words (2500) recur by chance."""

SIGMAS = 6
"""Word rows a training batch reserves above its expected word count, in standard deviations of that count: a
normal tail of ~1e-9 per step, so a 40-epoch run overflows with odds ~1e-4 (and an overflow only drops words)."""
ROW_ALIGN = 8
"""Row counts rounded up to a multiple of this, the tensor-core tile."""


class Windows(Protocol):
    """What a split needs of the holdout: its row metadata and its MEG windows in that order."""

    def metadata_frame(self) -> pd.DataFrame: ...

    def iter_windows(self, _batch_size: int, /) -> Iterable[tuple[np.ndarray, object]]: ...


@dataclass(frozen=True)
class Batch:
    """Spans packed into a fixed number of word rows, so every training step has the same shapes.

    `slot` is each span position's row of `x`, `len(x)` where the position is empty; `index` its split word.
    """

    x: Float[Tensor, "r c t"]
    onset: Float[Tensor, "g w"]
    slot: Int[Tensor, "g w"]
    index: Int[Tensor, "g w"]

    def target(self, labels: Int[Tensor, " n"]) -> Int[Tensor, "g w"]:
        """Each position's label from the split's `labels`; `IGNORE` where the position is empty."""
        return torch.where(self.slot < len(self.x), labels[self.index.clamp(min=0)], IGNORE)


@dataclass(frozen=True)
class Split:
    """Word windows on the device, with the spans that group them and each word's sampling weight."""

    x: Float[Tensor, "n c t"]
    y: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]
    onset: Float[Tensor, "g w"]
    weight: Float[Tensor, " n"]

    @classmethod
    def from_arrays(
        cls, x: Float[np.ndarray, "n c t"], y: Int[np.ndarray, " n"], spans: Spans, device: torch.device
    ) -> "Split":
        return cls.on_device(torch.from_numpy(x).to(device), y, spans, np.ones(len(y)))

    @classmethod
    def on_device(
        cls, x: Float[Tensor, "n c t"], y: Int[np.ndarray, " n"], spans: Spans, weight: Float[np.ndarray, " n"]
    ) -> "Split":
        device = x.device
        return cls(
            x,
            torch.from_numpy(y).to(device),
            torch.from_numpy(spans.index).to(device),
            torch.from_numpy(spans.onset).float().to(device),
            torch.from_numpy(weight).float().to(device),
        )

    @classmethod
    def gather(cls, runs: Iterable[tuple[Epochs, float]], device: torch.device) -> "Split":
        """Runs, each with its sampling weight, streamed to the device one at a time.

        Host memory never holds the whole split. Onsets restart with every run, so no span crosses two.
        """
        xs: list[Tensor] = []
        ys: list[np.ndarray] = []
        onsets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for run, weight in runs:
            xs.append(torch.from_numpy(run.x).to(device))
            ys.append(run.y)
            onsets.append(run.onset)
            weights.append(np.full(len(run.y), weight))
        spans = Spans.by_gap(np.concatenate(onsets))
        x = torch.cat(xs)
        xs.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cls.on_device(x, np.concatenate(ys), spans, np.concatenate(weights))

    @classmethod
    def validation(cls, epocher: Epocher, runs: list[Run], device: torch.device) -> dict[str, "Split"]:
        """The official validation runs, one split per task so each domain is scored alone."""
        held = [r for r in runs if r.partition is Partition.VALIDATION]
        return {
            task: cls.gather(((epocher.cached(r), 1.0) for r in held if r.key[0] == task), device)
            for task in sorted({r.key[0] for r in held})
        }

    @classmethod
    def from_holdout(cls, holdout: Windows, device: torch.device) -> "Split":
        """Every holdout window, grouped by sentence; isolated word rows are spans of one."""
        frame = holdout.metadata_frame()
        sentence = frame["source"] == "sentence"
        group = np.where(sentence, frame["epoch"], -1 - frame["index"])
        keys = frame.assign(group=group)[["subject", "source", "group"]]
        window = Window()
        x = np.concatenate([window(batch) for batch, _ in holdout.iter_windows(EVAL_SPANS)])
        spans = Spans.by_key(keys, frame["onset_s"].to_numpy())
        return cls.from_arrays(x, np.zeros(len(x), dtype=np.int64), spans, device)

    def log_prior(self) -> Float[Tensor, " v"]:
        """Label prior as training samples it, so logit adjustment undoes the same distribution."""
        return self.prior(self.y, self.weight)

    @staticmethod
    def prior(y: Int[Tensor, " n"], weight: Float[Tensor, " n"]) -> Float[Tensor, " v"]:
        counts = torch.bincount(y, weight, minlength=WORDS).float() + 1
        return (counts / counts.sum()).log()

    def batch(self, spans: Int[Tensor, " b"], rows: int) -> Batch:
        """The words of `spans` in `rows` rows, in span order, without a host sync; words past `rows` are dropped."""
        index = self.index[spans]
        order = (index != PAD).flatten().cumsum(0).view_as(index) - 1
        slot = torch.where((index != PAD) & (order < rows), order, rows)
        index = torch.where(slot < rows, index, PAD)
        words = index.new_zeros(rows + 1).scatter_(0, slot.flatten(), index.clamp(min=0).flatten())[:rows]
        return Batch(self.x[words], self.onset[spans], slot, index)

    def capacity(self, batch: int) -> int:
        """Word rows for `batch` spans drawn as training draws them: the expected count plus `SIGMAS` deviations."""
        lengths = (self.index != PAD).sum(dim=1).double()
        draw = self.weight[self.index[:, 0]].double()
        draw = draw / draw.sum()
        mean = float((draw * lengths).sum())
        spread = math.sqrt(batch * float((draw * (lengths - mean).square()).sum()))
        rows = ROW_ALIGN * math.ceil((batch * mean + SIGMAS * spread) / ROW_ALIGN)
        return min(rows, batch * self.index.shape[1])

    @staticmethod
    def seen(labelled: Iterable[tuple[Int[np.ndarray, " n"], Float[np.ndarray, " n"]]]) -> frozenset[tuple[int, ...]]:
        """Every span's label sequence across runs of (labels, onsets) — the word orders training has heard."""
        sequences: set[tuple[int, ...]] = set()
        for y, onset in labelled:
            index = Spans.by_gap(onset).index
            sequences.update(tuple(y[row[row != PAD]].tolist()) for row in index)
        return frozenset(sequences)

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[Tensor, " k"]:
        """Words in multi-word spans whose label sequence training never heard.

        The holdout's sentences are new stimuli, so a span recited from training measures memorised word order.
        """
        index = self.index.cpu().numpy()
        labels = self.y.cpu().numpy()
        rows = [row[row != PAD] for row in index]
        keep = [row for row in rows if len(row) > 1 and tuple(labels[row].tolist()) not in seen]
        return torch.from_numpy(np.concatenate([np.empty(0, np.int64), *keep])).to(self.y.device)

    def recurring(self, power: float) -> "Split":
        """Each recited span's words reweighted by how often training repeats its label sequence, to the `-power`."""
        index = self.index.cpu().numpy()
        labels = self.y.cpu().numpy()
        rows = [row[row != PAD] for row in index]
        sequences = [tuple(labels[row].tolist()) for row in rows]
        counts = Counter(sequences)
        scale = np.ones(len(labels))
        for row, sequence in zip(rows, sequences, strict=True):
            scale[row] = counts[sequence] ** -power if len(row) >= RECITED else 1.0
        return replace(self, weight=self.weight * torch.from_numpy(scale).float().to(self.weight.device))

    def dense(self) -> Int[Tensor, " d"]:
        """Words in spans at least `DENSE` long — the holdout-like subset."""
        long = self.index[(self.index != PAD).sum(dim=1) >= DENSE]
        return long[long != PAD]

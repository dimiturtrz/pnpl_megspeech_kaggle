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

from megspeech.data.spans import PAD, Spans
from megspeech.data.vocabulary import IGNORE, WORDS
from megspeech.data.windows import Windows

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


class Holdout(Protocol):
    """What a split needs of the holdout: its row metadata, in canonical row order."""

    def metadata_frame(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class Batch:
    """Spans packed into a fixed number of word rows, so every training step has the same shapes.

    `slot` is each span position's row of `x`, `len(x)` where the position is empty; `index` its split word.
    """

    x: Float[Tensor, "r c t"]
    onset: Float[Tensor, "g w"]
    slot: Int[Tensor, "g w"]
    index: Int[Tensor, "g w"]
    subject: Int[Tensor, " r"]

    def target(self, labels: Int[Tensor, " n"]) -> Int[Tensor, "g w"]:
        """Each position's label from the split's `labels`; `IGNORE` where the position is empty."""
        return torch.where(self.slot < len(self.x), labels[self.index.clamp(min=0)], IGNORE)


@dataclass(frozen=True)
class Split:
    """Word windows on the device, with the spans that group them and each word's sampling weight.

    A word labelled `IGNORE` is outside the vocabulary: its window feeds its span's context, but nothing scores it.
    `token` names each word itself, in or out of the vocabulary, so a span's token sequence is its sentence text.
    """

    windows: Windows
    y: Int[Tensor, " n"]
    token: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]
    onset: Float[Tensor, "g w"]
    weight: Float[Tensor, " n"]
    subject: Int[Tensor, " n"]

    @classmethod
    def from_arrays(
        cls, x: Float[np.ndarray, "n c t"], y: Int[np.ndarray, " n"], spans: Spans, device: torch.device
    ) -> "Split":
        return cls.of(Windows.stacked(torch.from_numpy(x).to(device)), (y, y), spans, np.ones(len(y)))

    @classmethod
    def of(
        cls,
        windows: Windows,
        words: tuple[Int[np.ndarray, " n"], Int[np.ndarray, " n"]],
        spans: Spans,
        weight: Float[np.ndarray, " n"],
        subject: Int[np.ndarray, " n"] | None = None,
    ) -> "Split":
        """A split of `windows` whose `words` are (label, token) per window, heard by `subject` (all 0 when None)."""
        device = windows.device
        y, token = words
        return cls(
            windows,
            torch.tensor(y, device=device),
            torch.tensor(token, device=device),
            torch.from_numpy(spans.index).to(device),
            torch.from_numpy(spans.onset).float().to(device),
            torch.tensor(weight, dtype=torch.float32, device=device),
            torch.tensor(np.zeros(len(y), dtype=np.int64) if subject is None else subject, device=device),
        )

    @classmethod
    def from_holdout(cls, holdout: Holdout, windows: Windows) -> "Split":
        """Every holdout row, its window the same row of `windows`, grouped by sentence; a lone word is its own span."""
        frame = holdout.metadata_frame()
        if len(frame) != len(windows):
            raise ValueError(f"{len(windows)} holdout windows for {len(frame)} holdout rows")
        sentence = frame["source"] == "sentence"
        group = np.where(sentence, frame["epoch"], -1 - frame["index"])
        keys = frame.assign(group=group)[["subject", "source", "group"]]
        spans = Spans.by_key(keys, frame["onset_s"].to_numpy())
        return cls.of(
            windows,
            (np.zeros(len(frame), dtype=np.int64),) * 2,
            spans,
            np.ones(len(frame)),
            frame["subject"].to_numpy(),
        )

    def sample(self, count: int) -> Float[Tensor, "k c t"]:
        """Up to `count` windows drawn without replacement."""
        return self.windows(torch.randperm(len(self.y), device=self.y.device)[:count])

    def labelled(self) -> Int[Tensor, " k"]:
        """Words with a vocabulary label — the ones a score counts."""
        return torch.nonzero(self.y != IGNORE).flatten()

    def log_prior(self) -> Float[Tensor, " v"]:
        """Label prior as training samples it, so logit adjustment undoes the same distribution."""
        return self.prior(self.y, self.weight)

    @staticmethod
    def prior(y: Int[Tensor, " n"], weight: Float[Tensor, " n"]) -> Float[Tensor, " v"]:
        known = y != IGNORE
        counts = torch.bincount(y[known], weight[known], minlength=WORDS).float() + 1
        return (counts / counts.sum()).log()

    def batch(self, spans: Int[Tensor, " b"], rows: int, jitter: int = 0) -> Batch:
        """The words of `spans` in `rows` rows, in span order, without a host sync; words past `rows` are dropped.
        Each window's start moves by up to `jitter` samples (`Windows.__call__`)."""
        index = self.index[spans]
        order = (index != PAD).flatten().cumsum(0).view_as(index) - 1
        slot = torch.where((index != PAD) & (order < rows), order, rows)
        index = torch.where(slot < rows, index, PAD)
        words = index.new_zeros(rows + 1).scatter_(0, slot.flatten(), index.clamp(min=0).flatten())[:rows]
        return Batch(self.windows(words, jitter), self.onset[spans], slot, index, self.subject[words])

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
    def seen(runs: Iterable[tuple[Int[np.ndarray, " n"], Spans]]) -> frozenset[tuple[int, ...]]:
        """Every span's token sequence across runs of (tokens, spans) — the sentences training has heard."""
        sequences: set[tuple[int, ...]] = set()
        for token, spans in runs:
            sequences.update(tuple(token[row[row != PAD]].tolist()) for row in spans.index)
        return frozenset(sequences)

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[Tensor, " k"]:
        """Labelled words in multi-word spans whose text training never heard.

        The holdout's sentences are new stimuli, so a span recited from training measures memorised word order. Text,
        not labels: under sentence spans every out-of-vocabulary word labels alike, so distinct sentences would collide.
        """
        index = self.index.cpu().numpy()
        token = self.token.cpu().numpy()
        rows = [row[row != PAD] for row in index]
        keep = [row for row in rows if len(row) > 1 and tuple(token[row].tolist()) not in seen]
        novel = np.concatenate([np.empty(0, np.int64), *keep])
        return torch.from_numpy(novel[self.y.cpu().numpy()[novel] != IGNORE]).to(self.y.device)

    def recurring(self, power: float) -> "Split":
        """Each recited span's words reweighted by how often training repeats its text, to the `-power`."""
        index = self.index.cpu().numpy()
        token = self.token.cpu().numpy()
        rows = [row[row != PAD] for row in index]
        sequences = [tuple(token[row].tolist()) for row in rows]
        counts = Counter(sequences)
        scale = np.ones(len(token))
        for row, sequence in zip(rows, sequences, strict=True):
            scale[row] = counts[sequence] ** -power if len(row) >= RECITED else 1.0
        return replace(self, weight=self.weight * torch.from_numpy(scale).float().to(self.weight.device))

    def dense(self) -> Int[Tensor, " d"]:
        """Labelled words in spans at least `DENSE` long — the holdout-like subset."""
        long = self.index[(self.index != PAD).sum(dim=1) >= DENSE]
        words = long[long != PAD]
        return words[self.y[words] != IGNORE]

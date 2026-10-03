"""Word windows on the device, grouped into spans, and the fixed-shape batches training and scoring draw from them."""

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.batch import Batch
from megspeech.data.batching.span_text import SpanText
from megspeech.data.batching.windows import Windows
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD, Spans

EVAL_SPANS = 256

DENSE = 4
"""Validation spans this long or longer look like holdout sentences (94% of which carry 4-9 words)."""

SIGMAS = 6
"""Word rows a training batch reserves above its expected word count, in standard deviations of that count: a
normal tail of ~1e-9 per step, so a 40-epoch run overflows with odds ~1e-4 (and an overflow only drops words)."""
ROW_ALIGN = 8
"""Row counts rounded up to a multiple of this, the tensor-core tile."""


class Holdout(Protocol):
    """What a split needs of the holdout: its row metadata, in canonical row order."""

    def metadata_frame(self) -> pd.DataFrame: ...


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
        counts = torch.bincount(y[known], weight[known], minlength=VOCAB_SIZE).float() + 1
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

    @property
    def text(self) -> SpanText:
        return SpanText(self.index.cpu().numpy(), self.token.cpu().numpy())

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[Tensor, " k"]:
        """Labelled words in multi-word spans whose text training never heard.

        The holdout's sentences are new stimuli, so a span recited from training measures memorised word order. Text,
        not labels: under sentence spans every out-of-vocabulary word labels alike, so distinct sentences would collide.
        """
        novel = self.text.novel(seen)
        return torch.from_numpy(novel[self.y.cpu().numpy()[novel] != IGNORE]).to(self.y.device)

    def recurring(self, power: float) -> "Split":
        """Each recited span's words reweighted by how often training repeats its text, to the `-power`."""
        scale = self.text.repeats(power)
        return replace(self, weight=self.weight * torch.from_numpy(scale).float().to(self.weight.device))

    def holdout_like(self) -> Int[Tensor, " d"]:
        """Labelled words in spans at least `DENSE` long — the holdout-like subset."""
        long = self.index[(self.index != PAD).sum(dim=1) >= DENSE]
        words = long[long != PAD]
        return words[self.y[words] != IGNORE]

"""Word windows on the device, grouped into spans, and the fixed-shape batches training and scoring draw from them.

A frozen encoder's per-window features (`megspeech.features.extract`) can stand in for a split's windows row for row:
each split's are saved in its own words' order.
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.windows import Windows
from megspeech.data.constants import RATE, TAIL
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD, Spans

BATCH = 64
"""Spans one training step reads by default."""

EVAL_SPANS = 256
"""Spans one evaluation forward reads beside a training step of `BATCH` spans."""

DENSE = 4
"""Validation spans this long or longer look like Kaggle test sentences (94% of which carry 4-9 words)."""

SIGMAS = 6
"""Word rows a training batch reserves above its expected word count, in standard deviations of that count: a
normal tail of ~1e-9 per step, so a 40-epoch run overflows with odds ~1e-4 (and an overflow only drops words)."""
ROW_ALIGN = 8
"""Row counts rounded up to a multiple of this, the tensor-core tile."""


@dataclass(frozen=True)
class SpanInput:
    """What a span model reads: word windows `x`, each span position's `onset` and row of `x` (`slot`, `len(x)` where
    the position is empty), each row's `subject`, and each span's `finish` (`Split.finish`)."""

    x: Float[Tensor, "r c t"]
    onset: Float[Tensor, "g w"]
    slot: Int[Tensor, "g w"]
    subject: Int[Tensor, " r"]
    finish: Float[Tensor, " g"] | None = None
    run: Int[Tensor, " r"] | None = None
    """Each row's run, numbered within its split; None reads every row as one run."""

    @property
    def mask(self) -> Bool[Tensor, "g w"]:
        """The span positions holding a word."""
        return self.slot < len(self.x)

    @property
    def runs(self) -> Int[Tensor, "g w"]:
        """Each span position's run, -1 where the position is empty."""
        rows = self.subject.new_zeros(len(self.x)) if self.run is None else self.run
        return torch.nn.functional.pad(rows, (0, 1), value=-1)[self.slot]

    def reading(self, x: Float[Tensor, "r c t"]) -> "SpanInput":
        """These spans over the windows `x` instead, row for row."""
        return replace(self, x=x)

    def timed(self, onset: Float[Tensor, "g w"]) -> "SpanInput":
        """These spans with their positions at `onset` instead."""
        return replace(self, onset=onset)


@dataclass(frozen=True)
class Batch(SpanInput):
    """Spans packed into a fixed number of word rows, so every training step has the same shapes: what a model reads
    of them, and `index`, each span position's split word (`PAD` where the position is empty)."""

    index: Int[Tensor, "g w"] = field(kw_only=True)

    def inputs(self, x: Float[Tensor, "r c t"] | None = None) -> SpanInput:
        """What a model reads of this batch, over the windows `x` when given (an augmented copy of its own)."""
        return SpanInput(self.x if x is None else x, self.onset, self.slot, self.subject, self.finish, self.run)

    def words_at(self, empty: int) -> Int[Tensor, "g w"]:
        """Each span position's split word, `empty` where the position holds none."""
        return torch.where(self.index != PAD, self.index, empty)

    def target(self, labels: Int[Tensor, " n"]) -> Int[Tensor, "g w"]:
        """Each position's label from the split's `labels`; `IGNORE` where the position is empty."""
        return torch.where(self.mask, labels[self.index.clamp(min=0)], IGNORE)


RECITED = 3
"""Spans this long recur only when a sentence is replayed; 2-word pairs of 50 words (2500) recur by chance."""


@dataclass(frozen=True)
class SpanText:
    """Span `g`'s words are `index[g]` (`PAD` past its end); word `i` reads as `token[i]`."""

    index: Int[np.ndarray, "g w"]
    token: Int[np.ndarray, " n"]

    def sequences(self) -> list[tuple[int, ...]]:
        """Every span's token sequence, in span order."""
        return [tuple(self.token[row[row != PAD]].tolist()) for row in self.index]

    def repeats(self, power: float) -> Float[np.ndarray, " n"]:
        """Each word's weight from how often its span's text recurs among the spans, to the `-power`; words of spans
        shorter than `RECITED`, and words in no span, keep 1."""
        packed = np.take_along_axis(self.index, np.argsort(self.index == PAD, axis=1, kind="stable"), axis=1)
        valid = packed != PAD
        keyed = np.c_[valid.sum(axis=1), np.where(valid, self.token[packed.clip(min=0)], 0)]
        _, inverse, counts = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
        recited = valid.sum(axis=1) >= RECITED
        span = np.where(recited, counts[inverse.reshape(-1)].astype(np.float64) ** -power, 1.0)
        scale = np.ones(len(self.token))
        scale[packed[valid]] = np.broadcast_to(span[:, None], packed.shape)[valid]
        return scale

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[np.ndarray, " k"]:
        """Words of multi-word spans whose text is not in `seen`, span by span."""
        unheard = np.array([sequence not in seen for sequence in self.sequences()], dtype=bool)
        rows = self.index[((self.index != PAD).sum(axis=1) > 1) & unheard]
        return rows[rows != PAD]


class Kaggle(Protocol):
    """What a split needs of the Kaggle test: its row metadata, in canonical row order."""

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
    source: Int[Tensor, " n"] | None = None
    """Each word's recording source, an index into the sorted sources of the split; None where unknown."""
    read: Bool[Tensor, " n"] | None = None
    """Whether each word is of a read-sentence run (`DataConfig.reads`); None where unknown."""
    muted: Bool[Tensor, " n"] | None = None
    """Whether each word's label is kept from the context head (`DataConfig.muted`); None mutes none."""
    finish: Float[Tensor, " g"] | None = None
    """Each span's last word's end, in seconds from its first onset: its epoch's bound less `TAIL`, as a Kaggle test
    sentence epoch's length less `sentence_extra_end_s` gives it; None where unknown (`ended`)."""
    run: Int[Tensor, " n"] | None = None
    """Each word's run (a Kaggle test archive), numbered from 0 within the split; None where unknown."""

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

    def finished(self, rate: float = RATE) -> "Split":
        """This split with each span's `finish`, its windows sampled at `rate`."""
        return replace(self, finish=self.ended(self.windows, self.index, self.onset, rate))

    @staticmethod
    def ended(
        windows: Windows,
        index: Int[Tensor, "g w"],
        onset: Float[Tensor, "g w"],
        rate: float = RATE,
        tail: float = TAIL,
    ) -> Float[Tensor, " g"]:
        """Each span's last word's end in seconds from its first onset: its next-onset bound, which past a span's last
        word is its epoch's end, less `tail` (the epoch's `TAIL`; 0 where the bound is the next word's onset, as for a
        span cut short of its sentence's end). Without next onsets every end is 0, before the last onset: unknown."""
        if windows.following is None:
            return onset.new_zeros(len(index))
        place = ((index != PAD).sum(dim=1) - 1).clamp(min=0)[:, None]
        last = index.gather(1, place)[:, 0].clamp(min=0)
        heard = (windows.following[last] - windows.at[last]).to(onset.dtype) / rate - tail
        return onset.gather(1, place)[:, 0] + heard

    @classmethod
    def from_kaggle(cls, kaggle: Kaggle, windows: Windows, rate: float = RATE) -> "Split":
        """Every Kaggle test row, its window the same row of `windows` (at `rate`), grouped by sentence; a lone word is
        its own span."""
        frame = kaggle.metadata_frame()
        if len(frame) != len(windows):
            raise ValueError(f"{len(windows)} Kaggle test windows for {len(frame)} Kaggle test rows")
        sentence = frame["source"] == "sentence"
        group = np.where(sentence, frame["epoch"], -1 - frame["index"])
        keys = frame.assign(group=group)[["subject", "source", "group"]]
        spans = Spans.by_key(keys, frame["onset_s"].to_numpy())
        held = cls.of(
            windows,
            (np.zeros(len(frame), dtype=np.int64),) * 2,
            spans,
            np.ones(len(frame)),
            frame["subject"].to_numpy(),
        ).finished(rate)
        return held.numbered(pd.MultiIndex.from_frame(frame[["subject", "source"]]))

    def numbered(self, runs: "pd.Series[str] | pd.Index[str]") -> "Split":
        """This split with each word's run named by `runs`, numbered from 0 in order of appearance."""
        return replace(self, run=torch.tensor(pd.factorize(runs)[0].astype(np.int64), device=self.y.device))

    def runs(self) -> int:
        """How many runs the split's words come from; 0 where unknown."""
        return 0 if self.run is None else int(self.run.max()) + 1

    def runs_of(self, words: Int[Tensor, " r"]) -> Int[Tensor, " r"] | None:
        """The run of each of `words`; None where unknown."""
        return None if self.run is None else self.run[words]

    def sample(self, count: int) -> Float[Tensor, "k c t"]:
        """Up to `count` windows drawn without replacement."""
        return self.windows(self.drawn(count))

    def drawn(self, count: int) -> Int[Tensor, " k"]:
        """Up to `count` words drawn without replacement."""
        return torch.randperm(len(self.y), device=self.y.device)[:count]

    def labelled(self) -> Int[Tensor, " k"]:
        """Words with a vocabulary label — the ones a score counts."""
        return torch.nonzero(self.y != IGNORE).flatten()

    def unmuted(self, labels: Int[Tensor, " n"]) -> Int[Tensor, " n"]:
        """`labels` of the words the context head may train on: `IGNORE` where a word is `muted`."""
        return labels if self.muted is None else torch.where(self.muted, IGNORE, labels)

    def reads(self) -> Bool[Tensor, " g"]:
        """Each span's read flag, from its first word's; no span is read where `read` is unknown."""
        if self.read is None:
            return torch.zeros(len(self.index), dtype=torch.bool, device=self.y.device)
        return self.read[self.index[:, 0]]

    def lone(self) -> Bool[Tensor, " n"]:
        """Words alone in their span, as the Kaggle test's isolated-word rows are: no context reaches them."""
        alone = (self.index != PAD).sum(dim=1) == 1
        mask = torch.zeros(len(self.y), dtype=torch.bool, device=self.y.device)
        mask[self.index[alone, 0]] = True
        return mask

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
        Each window's start moves by up to `jitter` samples (`Windows.__call__`). Windows with a `segment` batch one row
        per span instead (`segmented`)."""
        if self.windows.segment:
            return self.segmented(spans)
        index = self.index[spans]
        order = (index != PAD).flatten().cumsum(0).view_as(index) - 1
        slot = torch.where((index != PAD) & (order < rows), order, rows)
        index = torch.where(slot < rows, index, PAD)
        words = index.new_zeros(rows + 1).scatter_(0, slot.flatten(), index.clamp(min=0).flatten())[:rows]
        return Batch(
            self.windows(words, jitter),
            self.onset[spans],
            slot,
            self.subject[words],
            self.ends(spans),
            self.runs_of(words),
            index=index,
        )

    def segmented(self, spans: Int[Tensor, " b"]) -> Batch:
        """One row per span: its continuous segment from its first onset, `slot` naming that row at each of the span's
        words. Never jittered, since every word's place in the segment is its onset."""
        index = self.index[spans]
        first = index[:, 0]
        segments = replace(self.windows, length=self.windows.segment, offset=0, baseline=0)
        rows = torch.arange(len(spans), device=index.device)[:, None].expand_as(index)
        slot = torch.where(index != PAD, rows, len(spans))
        return Batch(
            segments(first),
            self.onset[spans],
            slot,
            self.subject[first],
            self.ends(spans),
            self.runs_of(first),
            index=index,
        )

    def ends(self, spans: Int[Tensor, " b"]) -> Float[Tensor, " b"] | None:
        """`finish` of `spans`; None where unknown."""
        return None if self.finish is None else self.finish[spans]

    def capacity(self, batch: int, weights: Float[Tensor, " g"]) -> int:
        """Word rows for `batch` spans drawn at `weights`: the expected count plus `SIGMAS` deviations."""
        lengths = (self.index != PAD).sum(dim=1).double()
        draw = weights.double() / weights.double().sum()
        mean = float((draw * lengths).sum())
        spread = math.sqrt(batch * float((draw * (lengths - mean).square()).sum()))
        rows = ROW_ALIGN * math.ceil((batch * mean + SIGMAS * spread) / ROW_ALIGN)
        return min(rows, batch * self.index.shape[1])

    @property
    def text(self) -> SpanText:
        return SpanText(self.index.cpu().numpy(), self.token.cpu().numpy())

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[Tensor, " k"]:
        """Labelled words in multi-word spans whose text training never heard.

        Kaggle test sentences are new stimuli, so a span recited from training measures memorised word order. Text,
        not labels: under sentence spans every out-of-vocabulary word labels alike, so distinct sentences would collide.
        """
        novel = self.text.novel(seen)
        return torch.from_numpy(novel[self.y.cpu().numpy()[novel] != IGNORE]).to(self.y.device)

    def recurring(self, power: float) -> "Split":
        """Each recited span's words reweighted by how often training repeats its text, to the `-power`."""
        scale = self.text.repeats(power)
        return replace(self, weight=self.weight * torch.from_numpy(scale).float().to(self.weight.device))

    def spanned(self, spans: Int[Tensor, " b"]) -> "Split":
        """This split reduced to `spans`: words keep their ids, so only the words `heard` names are read."""
        return replace(self, index=self.index[spans], onset=self.onset[spans], finish=self.ends(spans))

    def heard(self) -> Int[Tensor, " k"]:
        """Labelled words that some span of this split reads."""
        words = self.index[self.index != PAD]
        return words[self.y[words] != IGNORE]

    def permuted(self, seed: int) -> "Split":
        """This split with every word's window (start, end, first readable sample, next onset) another word's, onsets
        untouched: what a model's timing alone reads."""
        order = torch.randperm(len(self.y), generator=torch.Generator().manual_seed(seed)).to(self.y.device)
        windows = self.windows
        moved = {
            name: getattr(windows, name)[order]
            for name in ("at", "end", "begin", "following")
            if getattr(windows, name) is not None
        }
        return replace(self, windows=replace(windows, **moved))

    def ordered(self, seed: int) -> "Split":
        """This split with each span's words dealt among its own word slots in a random order, onsets untouched: every
        word keeps its window and label but meets another position's timing and other neighbours, what a model's
        memory of sentence text reads."""
        valid = self.index != PAD
        keys = torch.rand(self.index.shape, generator=torch.Generator().manual_seed(seed)).to(self.index.device)
        slots = (~valid).int().argsort(dim=1, stable=True)
        dealt = self.index.gather(1, torch.where(valid, keys, 2.0).argsort(dim=1))
        return replace(self, index=self.index.scatter(1, slots, dealt))

    def kaggle_like(self) -> Int[Tensor, " d"]:
        """Labelled words in spans at least `DENSE` long — the Kaggle-test-like subset."""
        long = self.index[(self.index != PAD).sum(dim=1) >= DENSE]
        words = long[long != PAD]
        return words[self.y[words] != IGNORE]


FEATURES = "features"
JOIN = "+"
TRAIN = "train"
KAGGLE_FEATURES = "holdout"
"""The names of the training split's features and the prefix of a track's Kaggle test's, `holdout-<track>`; a
validation domain's are its own name."""


@dataclass(frozen=True)
class Features:
    """The features saved under `directory`, one file per split; no directory reads every split's MEG."""

    directory: Path | None = None
    joined: tuple["Features", ...] = ()
    """Features of other encoders read beside these, word for word: each expert's embedding one stretch of channels."""

    @classmethod
    def under(cls, root: DataRoot, name: str) -> "Features":
        """The features `name` names; `a+b` joins `b`'s after `a`'s."""
        if not name:
            return cls()
        first, *rest = (root.kind("models") / FEATURES / part for part in name.split(JOIN))
        return cls(first, tuple(cls(directory) for directory in rest))

    def path(self, split: str) -> Path:
        if self.directory is None:
            raise ValueError("no features named")
        return self.directory / f"{split}.pt"

    def save(self, split: str, features: Float[Tensor, "n c *k"]) -> Path:
        path = self.path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features.cpu(), path)
        return path

    def read(self, split: str, windows: Windows) -> Windows:
        """`windows` itself, or the saved features of `split` as its windows, uncentred: channels by
        features each, one feature when saved flat."""
        if self.directory is None:
            return windows
        parts = [self.loaded(split, windows), *(other.loaded(split, windows) for other in self.joined)]
        return Windows.stacked(torch.cat(parts, dim=1), centred=False)

    def loaded(self, split: str, windows: Windows) -> Float[Tensor, "n c k"]:
        """The saved features of `split`, one row per window of `windows`."""
        features = torch.load(self.path(split), map_location=windows.device, weights_only=True)
        if len(features) != len(windows):
            raise ValueError(f"{len(features)} {split} features for {len(windows)} windows")
        return features.reshape(len(features), features.shape[1], -1)

    def split(self, name: str, split: Split) -> Split:
        """`split` reading `name`'s features as its windows."""
        return replace(split, windows=self.read(name, split.windows))

"""How much of a local test score is sentence memorisation: each test word's hit, split by whether its sentence's text
occurs verbatim in the training transcripts and by how long a run of its span's words training repeats.

`Heard` holds the whole sentences and every n-gram of the runs the model trained on; `Heard.tags` marks each word of a
scored split. `Tally` reads the buckets of the tagged words per source, as balanced accuracy (the repo's macro over
classes, with its bootstrap interval) and as the micro top-k hit rate, steadier in small buckets (megspeech-7gf.111).

    uv run python -m megspeech.submission sent40-ioi-ema9998-s0 --out x --local --memorisation --grouping kaggle
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, Vocabulary
from megspeech.data.raw.runs import Column, Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.metric.ruler import Ruler

LOG = logging.getLogger(__name__)

MAX_RUN = 12
"""Longest n-gram searched for in training: past a span's nine words no run can be longer."""
GRAM = 3
MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)
POOLED = "pooled"
SHORT_RUN = 2
LONG_RUN = 5


@dataclass(frozen=True)
class NGrams:
    """The hash of every n-gram, n up to `MAX_RUN`, that occurs inside one sentence of the training text."""

    seen: dict[int, Int[np.ndarray, " m"]]

    @staticmethod
    def windows(
        tokens: Int[np.ndarray, " n"], group: Int[np.ndarray, " n"], n: int
    ) -> tuple[Int[np.ndarray, " m"], Int[np.ndarray, " m"]]:
        """Hash and group of every `n`-word window of `tokens` that lies inside one `group`."""
        count = len(tokens) - n + 1
        if count <= 0:
            return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
        words = tokens.astype(np.uint64)
        hashed = np.zeros(count, dtype=np.uint64)
        for step in range(n):
            hashed = hashed * MULTIPLIER + words[step : step + count]
        inside = group[:count] == group[n - 1 : n - 1 + count]
        return hashed[inside], group[:count][inside]

    @classmethod
    def of(cls, tokens: Int[np.ndarray, " n"], group: Int[np.ndarray, " n"]) -> "NGrams":
        return cls({n: np.unique(cls.windows(tokens, group, n)[0]) for n in range(1, MAX_RUN + 1)})

    def longest(
        self, tokens: Int[np.ndarray, " n"], group: Int[np.ndarray, " n"], groups: int
    ) -> Int[np.ndarray, " g"]:
        """Per group of `tokens`, the length of its longest run of words that training also holds in one sentence."""
        best = np.zeros(groups, dtype=np.int64)
        for n in range(1, MAX_RUN + 1):
            hashed, owner = self.windows(tokens, group, n)
            np.maximum.at(best, owner[np.isin(hashed, self.seen[n])], n)
        return best

    def coverage(
        self, tokens: Int[np.ndarray, " n"], group: Int[np.ndarray, " n"], groups: int
    ) -> Float[np.ndarray, " g"]:
        """Per group, the share of its `GRAM`-word windows training holds; NaN for a group shorter than `GRAM`."""
        hashed, owner = self.windows(tokens, group, GRAM)
        windows = np.bincount(owner, minlength=groups)
        found = np.bincount(owner, weights=np.isin(hashed, self.seen[GRAM]), minlength=groups)
        return np.where(windows > 0, found / np.maximum(windows, 1), np.nan)


@dataclass(frozen=True)
class Heard:
    """What training read: its sentences whole, and their n-grams."""

    sentences: frozenset[tuple[int, ...]]
    grams: NGrams

    @staticmethod
    def texts(
        table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[tuple[int, ...]], Int[np.ndarray, " n"], Int[np.ndarray, " n"]]:
        """`table`'s sentences: their (run, sentence) keys, token sequences, and each word's token and sentence code."""
        grouped = table.groupby([Column.RUN, Column.SENTENCE], sort=False)
        group, tokens = grouped.ngroup().to_numpy(), Vocabulary.token(table[Column.WORD])
        cuts = np.flatnonzero(np.diff(group)) + 1
        keys = grouped.size().reset_index()[[Column.RUN, Column.SENTENCE]]
        return keys, [tuple(text.tolist()) for text in np.split(tokens, cuts)], tokens, group

    @classmethod
    def of(cls, table: pd.DataFrame) -> "Heard":
        """The text of `table`'s words (`run`, `sentence`, `word`, in spoken order)."""
        _, texts, tokens, group = cls.texts(table)
        return cls(frozenset(texts), NGrams.of(tokens, group))

    def tags(self, split: Split, words: pd.DataFrame, spoken: pd.DataFrame) -> pd.DataFrame:
        """For every word of `split` (its rows `words`, from sentences of `spoken`), whether its whole sentence is a
        training sentence, the longest run of its span's words training repeats, and the share of the span's
        `GRAM`-word windows it holds."""
        keys, texts, _, _ = self.texts(spoken)
        keys = keys.assign(seen=[text in self.sentences for text in texts])
        seen = (
            words[[Column.RUN, Column.SENTENCE]]
            .reset_index(drop=True)
            .merge(keys, how="left")["seen"]
            .to_numpy(dtype=bool)
        )
        index = split.index.cpu().numpy()
        valid = index != PAD
        span = np.broadcast_to(np.arange(len(index))[:, None], index.shape)[valid]
        flat = index[valid]
        tokens = split.token.cpu().numpy()[flat]
        longest = np.zeros(len(split.y), dtype=np.int64)
        share = np.full(len(split.y), np.nan)
        longest[flat] = self.grams.longest(tokens, span, len(index))[span]
        share[flat] = self.grams.coverage(tokens, span, len(index))[span]
        return pd.DataFrame({"seen": seen, "longest": longest, "trigram": share})


class Bucket(StrEnum):
    """A subset of the test words; the sentence ones partition them, the run ones partition them by span overlap."""

    ALL = "all"
    SEEN = "seen-sentence"
    NOVEL = "novel-sentence"
    SCORED = "repo-scored"
    SHORT = "run<=2"
    MIDDLE = "run3-4"
    LONG = "run>=5"

    def mask(self, frame: pd.DataFrame) -> Bool[np.ndarray, " n"]:
        longest = frame["longest"]
        return {
            Bucket.ALL: pd.Series(np.ones(len(frame), dtype=bool), index=frame.index),
            Bucket.SEEN: frame["seen"],
            Bucket.NOVEL: ~frame["seen"],
            Bucket.SCORED: frame["scored"],
            Bucket.SHORT: longest <= SHORT_RUN,
            Bucket.MIDDLE: (longest > SHORT_RUN) & (longest < LONG_RUN),
            Bucket.LONG: longest >= LONG_RUN,
        }[self].to_numpy(dtype=bool)


@dataclass(frozen=True)
class Tally:
    """Test words, one row each: `source`, `label`, `hit` (in the top k), `seen`, `longest`, `trigram`, `scored`."""

    frame: pd.DataFrame
    ruler: Ruler = Ruler()

    def line(self, rows: pd.DataFrame) -> str:
        """Count, classes, balanced accuracy with its bootstrap interval, micro hit rate and mean trigram share."""
        hits = torch.tensor(rows["hit"].to_numpy(dtype=bool))
        labels = torch.tensor(rows["label"].to_numpy(dtype=np.int64))
        interval = self.ruler.interval(self.ruler.draws(hits, labels))
        return (
            f"n={len(rows)} classes={rows['label'].nunique()} bacc={self.ruler.balanced(hits, labels):.4f} "
            f"[{interval.low:.4f}, {interval.high:.4f}] micro={float(hits.float().mean()):.4f} "
            f"trigram={rows['trigram'].mean():.3f}"
        )

    def report(self) -> list[str]:
        """One line per source (and all sources pooled) and bucket that holds a word."""
        pooled = self.frame.assign(source=POOLED)
        lines: list[str] = []
        for source, rows in pd.concat([self.frame, pooled]).groupby("source", sort=False):
            lines += [
                f"{source} {bucket}: {self.line(rows[bucket.mask(rows)])}"
                for bucket in Bucket
                if bucket.mask(rows).any()
            ]
        return lines


@dataclass(frozen=True)
class Memorisation:
    """A loaded model scored on a partition's labelled words, each tagged by what training heard."""

    loaded: LoadedModel

    def words(self, runs: list[Run], tau: float) -> pd.DataFrame:
        """Every labelled word of the scored domains of `runs`: its hit at `tau` and its `Heard` tags."""
        held = self.loaded
        ruled = held.ruled(runs)
        scored = [run for group in ruled.values() for run in group]
        _, table = held.load(scored)
        spoken, _ = held.spoken(scored)
        trained = held.data.trained(Run.everywhere(held.root))
        heard = Heard.of(held.data.run_cache(held.root).table([run.name for run in trained]))
        splits, _ = held.domains(ruled)
        frames = []
        for name, split in splits.items():
            rows = table[table[Column.RUN].isin([run.name for run in ruled[name]])]
            context = held.evaluator.head(held.evaluator.logits(held.model, split))
            adjusted = held.calibration(tau).adjusted(context)
            hits = Ruler().hits(adjusted, split.y.clamp(min=0)).cpu().numpy()
            novel = np.isin(np.arange(len(split.y)), split.novel(held.evaluator.seen).cpu().numpy())
            tagged = heard.tags(split, rows, spoken).assign(
                source=name, label=split.y.cpu().numpy(), hit=hits, scored=novel
            )
            frames.append(tagged[split.y.cpu().numpy() != IGNORE])
        return pd.concat(frames, ignore_index=True)

    def report(self, runs: list[Run], tau: float, label: str) -> None:
        """Log `Tally`'s lines for the scored domains of `runs` at `tau`."""
        for line in Tally(self.words(runs, tau)).report():
            LOG.info("%s memorisation tau %s %s", label, tau, line)

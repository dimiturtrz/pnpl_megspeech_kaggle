"""Where a model's remaining loss sits among the 50 words (megspeech-7gf.106, megspeech-7gf.102).

Reads the saved `probs.pt` of every member of a blend, averages their context log-probabilities (the blend's own
geometric mean), finishes them at the tau that validation picks, and scores the local test's unseen-span words with the
selection ruler (`Scored`, the confusion study's ruler). No model is run; the data is read only for each word's place
in its span and its length. Per word: top-10 hit rate and count, by source, training prior, acoustic duration and
part of speech; the most frequent top-1 confusions and whether they stay inside a part of speech or a first
phoneme beyond what the predicted marginal alone would give; and the hit rate by position in the span and
inter-onset interval, both raw and as an excess over each word's own rate.

    uv run python -m megspeech.evaluation.studies errors.perword m11xsp7hub6d2v3
"""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int, Shaped
from scipy.stats import spearmanr
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import METRICS
from megspeech.data.labels.phonemes import EVENTS, UNKNOWN, Pronunciation
from megspeech.data.labels.syntax import TAGS, PartsOfSpeech
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary
from megspeech.data.raw.events import Event, Events, Kind
from megspeech.data.raw.runs import Column, Partition, Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device, Placed
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.metric.ruler import TOP
from megspeech.evaluation.studies.errors.confusion import Lengths
from megspeech.evaluation.studies.errors.crowding import TEST, VAL, Finishes, Saved
from megspeech.evaluation.studies.errors.scored import Scored
from megspeech.evaluation.studies.fusion.neighbour import DEEP

LOG = logging.getLogger(__name__)

PAIRS = 15
"""Most frequent top-1 confusions listed."""
EXTREMES = 10
"""Best and worst words listed."""
BUCKETS = 4
"""Inter-onset-interval quantile buckets."""
FUNCTION_TAGS = ("ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ")
"""Closed-class parts of speech: the words a transcript carries least in its content."""
OUT = "perword.json"


class Position(StrEnum):
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


@dataclass(frozen=True)
class Place:
    """Where each scored word sits: its slot in the span and its timing."""

    position: Shaped[np.ndarray, " n"]
    gap_next: Float[np.ndarray, " n"]
    gap_prev: Float[np.ndarray, " n"]
    span: Int[np.ndarray, " n"]
    """Each word's span, numbered from 0 and distinct across concatenated places."""

    @classmethod
    def of(cls, split: Split, ids: Int[Tensor, " k"]) -> "Place":
        """`ids`' rows of `split`: first, middle or last of their span, seconds to the next and from the previous
        onset (NaN at a span's end), and which span."""
        index, onset = split.index.cpu().numpy(), split.onset.cpu().numpy()
        valid = index != PAD
        slot = np.broadcast_to(np.arange(index.shape[1]), index.shape)[valid]
        length = np.broadcast_to(valid.sum(axis=1)[:, None], index.shape)[valid]
        place = np.where(slot == 0, Position.FIRST, np.where(slot == length - 1, Position.LAST, Position.MIDDLE))
        rows = len(split.y)
        position = np.full(rows, Position.MIDDLE.value, dtype=object)
        position[index[valid]] = place
        span = np.full(rows, -1)
        span[index[valid]] = np.broadcast_to(np.arange(index.shape[0])[:, None], index.shape)[valid]
        gap = onset[:, 1:] - onset[:, :-1]
        both = valid[:, 1:] & valid[:, :-1]
        following, preceding = np.full(rows, np.nan), np.full(rows, np.nan)
        following[index[:, :-1][both]] = gap[both]
        preceding[index[:, 1:][both]] = gap[both]
        keep = ids.cpu().numpy()
        return cls(position[keep], following[keep], preceding[keep], span[keep])

    def concat(self, other: "Place") -> "Place":
        return Place(
            np.concatenate([self.position, other.position]),
            np.concatenate([self.gap_next, other.gap_next]),
            np.concatenate([self.gap_prev, other.gap_prev]),
            np.concatenate([self.span, other.span + self.span.max(initial=-1) + 1]),
        )

    @classmethod
    def joined(cls, places: Sequence["Place"]) -> "Place":
        """`places` concatenated in order, their spans kept distinct."""
        joined = places[0]
        for place in places[1:]:
            joined = joined.concat(place)
        return joined


@dataclass(frozen=True)
class Located:
    """The local test's ruled runs by domain, each domain's split and the words the runs carried, in split order."""

    held: dict[str, list[Run]]
    splits: dict[str, Split]
    carried: pd.DataFrame

    def place(self, name: str, ids: Int[Tensor, " k"]) -> Place:
        return Place.of(self.splits[name], ids)

    def spoken(self, name: str, ids: Int[Tensor, " k"]) -> pd.DataFrame:
        """`ids`' rows of the words `name`'s split was cut from, checked word for word against the split."""
        runs = [run.name for run in self.held[name]]
        words = self.carried[self.carried[Column.RUN].isin(runs)].reset_index(drop=True)
        if not np.array_equal(words[Column.TOKEN].to_numpy(), self.splits[name].token.cpu().numpy()):
            raise ValueError("the carried words and the split disagree")
        return words.loc[ids.cpu().numpy(), [Column.RUN, Column.ONSET, Column.WORD]].reset_index(drop=True)

    def agrees(self, name: str, ids: Int[Tensor, " k"], labels: Int[Tensor, " k"]) -> bool:
        """Whether `name`'s split labels `ids`' words as `labels` does."""
        return torch.equal(self.splits[name].y[ids], labels)


@dataclass(frozen=True)
class Attributes:
    """Per-word facts a confusion can stay inside: part of speech, first phoneme, first two phonemes."""

    pos: Int[np.ndarray, " v"]
    first: Int[np.ndarray, " v"]
    first_two: Int[np.ndarray, " v"]
    duration: Float[np.ndarray, " v"]
    """Mean seconds the word is spoken for in the stimuli's forced alignment, NaN if it never occurs."""

    @staticmethod
    def durations(events: Sequence[Path]) -> Float[np.ndarray, " v"]:
        spoken = pd.concat(
            [Events(path).read(Event.DURATION) for path in events],
            ignore_index=True,
        )
        words = spoken[spoken[Event.KIND] == Kind.WORD]
        mean = words.groupby(Vocabulary.spelled(words[Event.SEGMENT]))[Event.DURATION].mean()
        return mean.reindex(list(PRIMARY.words)).to_numpy()

    @classmethod
    def of(cls, root: DataRoot) -> "Attributes":  # devtools-ignore: test-mirror
        tokens = torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words)))
        spelled = Pronunciation.load(Pronunciation.path(root)).phonemes(tokens).numpy()
        tags = PartsOfSpeech.load(PartsOfSpeech.path(root)).classes(tokens).numpy()
        events = [path for path in sorted(root.raw().rglob(EVENTS)) if path.stat().st_size]
        first_two = spelled[:, 0] * (spelled.max() + 1) + spelled[:, 1]
        return cls(tags, spelled[:, 0], first_two, cls.durations(events))

    def function(self) -> Bool[np.ndarray, " v"]:
        names = np.array([TAGS[tag] if tag != UNKNOWN else "" for tag in self.pos])
        return np.isin(names, FUNCTION_TAGS)

    def named(self) -> dict[str, Int[np.ndarray, " v"]]:
        return {"pos": self.pos, "first-phoneme": self.first, "first-two-phonemes": self.first_two}


@dataclass(frozen=True)
class Occurrences:
    """The scored words: ruler-scored predictions beside each word's place."""

    scored: Scored
    place: Place

    def hits(self) -> Bool[np.ndarray, " n"]:
        return self.scored.hits()

    def own_rate(self) -> Float[np.ndarray, " n"]:
        """Each word's hit rate over all its occurrences, so `excess` removes which word it is."""
        return self.scored.hit_rate()[self.scored.labels]

    def buckets(self, by: Shaped[np.ndarray, " n"]) -> pd.DataFrame:
        """Words, hit rate and mean excess over the word's own rate, per value of `by` (rows with no value dropped)."""
        frame = pd.DataFrame({"by": by, "hit": self.hits(), "excess": self.hits() - self.own_rate()}).dropna(
            subset="by"
        )
        grouped = frame.groupby("by", observed=True)
        return pd.DataFrame({"n": grouped.size(), "hit10": grouped["hit"].mean(), "excess": grouped["excess"].mean()})

    def by_position(self) -> pd.DataFrame:
        return self.buckets(self.place.position).reindex([position.value for position in Position])

    def by_interval(self, column: str) -> pd.DataFrame:
        """`buckets` of the quantile bucket of `gap_next` or `gap_prev`."""
        values = getattr(self.place, column)
        cut = pd.qcut(pd.Series(values), BUCKETS, duplicates="drop").astype(object).to_numpy()
        return self.buckets(cut).sort_index(key=lambda labels: labels.map(lambda interval: interval.left))

    def by_word(self, prior: Float[np.ndarray, " v"], attributes: Attributes) -> pd.DataFrame:
        """One row per word: count, top-10 hit rate overall and per source, top-1 recall, prior, acoustic duration, and
        mean seconds to the next onset."""
        scored, hits = self.scored, self.hits()
        columns = {
            "n": scored.support(),
            "hit10": scored.hit_rate(),
            "top1": scored.recall(),
            "prior": prior,
            "duration": attributes.duration,
            "ioi": Lengths.mean_by_target(scored.labels, self.place.gap_next),
            "function": attributes.function(),
        }
        for name in dict.fromkeys(scored.domains):
            here = scored.domains == name
            got = np.bincount(scored.labels[here], weights=hits[here], minlength=VOCAB_SIZE)
            held = np.bincount(scored.labels[here], minlength=VOCAB_SIZE)
            columns[f"{name}-n"] = held
            columns[f"{name}-hit10"] = np.where(held > 0, got / np.maximum(held, 1), np.nan)
        return pd.DataFrame(columns, index=list(PRIMARY.words))

    @staticmethod
    def correlations(table: pd.DataFrame) -> dict[str, float]:
        """Spearman of each word's hit rate against its log prior, duration and count (words with a hit rate)."""
        held = table.dropna(subset=["hit10", "duration", "ioi"])
        return {
            name: float(spearmanr(held["hit10"], column).statistic)
            for name, column in (
                ("log-prior", np.log(held["prior"])),
                ("duration", held["duration"]),
                ("ioi", held["ioi"]),
                ("n", held["n"]),
            )
        }


@dataclass(frozen=True)
class Confusions:
    """Top-1 errors among the scored words."""

    counts: Int[np.ndarray, "v v"]

    @classmethod
    def of(cls, scored: Scored) -> "Confusions":
        counts = scored.confusion()
        return cls(counts - np.diag(np.diag(counts)))

    def top(self, k: int) -> list[tuple[int, int, int]]:
        """The `k` most frequent (true, predicted, count) errors."""
        order = np.argsort(-self.counts, axis=None, kind="stable")[:k]
        rows, cols = np.unravel_index(order, self.counts.shape)
        return [(int(r), int(c), int(self.counts[r, c])) for r, c in zip(rows, cols, strict=True)]

    def kept(self, attribute: Int[np.ndarray, " v"]) -> tuple[float, float]:
        """Share of errors whose predicted word shares `attribute` with the true one, and the share expected if the
        predicted word were drawn from the errors' predicted marginal alone (never the true word itself)."""
        same = (attribute[:, None] == attribute[None, :]) & ~np.eye(VOCAB_SIZE, dtype=bool)
        observed = (self.counts * same).sum() / self.counts.sum()
        true, predicted = self.counts.sum(axis=1), self.counts.sum(axis=0)
        pairs = np.outer(true, predicted) * ~np.eye(VOCAB_SIZE, dtype=bool)
        return float(observed), float((pairs * same).sum() / pairs.sum())


@dataclass(frozen=True)
class Scene:
    """A blend's local-test occurrences at the tau validation chose."""

    occurrences: Occurrences
    prior: Float[np.ndarray, " v"]
    tau: float
    word: Scored | None = None
    """The same words on the members' word head (one window, no context), finished at its own val tau."""
    spoken: pd.DataFrame | None = None
    """Each scored word's `run`, `onset` and heard `word`, in the scored order."""

    @classmethod
    def of(cls, placed: Placed) -> "Scene":  # devtools-ignore: test-mirror
        root = placed.root
        members = json.loads((root.models() / placed.name / METRICS).read_text())["members"]
        directories = [root.models() / member for member in members]
        val, saved = Saved.load(directories, VAL), Saved.load(directories, TEST)
        finishes = Finishes.fit(val)
        LOG.info(
            "tau %.2f context, %.2f word head, fit on %s (%d members)",
            finishes.context.tau,
            finishes.word.tau,
            VAL,
            len(members),
        )
        test, ids = saved.domains(), saved.ids()
        loaded = replace(Loader.of(Group(root, DEEP, placed.device), members[:1]), partition=Partition.TEST)
        held = loaded.ruled(Run.discover(root.raw()))
        splits, _ = loaded.domains(held)
        _, carried = loaded.spoken([run for runs in held.values() for run in runs])
        located = Located(held, splits, carried)
        place = Place.joined([located.place(name, ids[name]) for name in test])
        spoken = pd.concat([located.spoken(name, ids[name]) for name in test], ignore_index=True)
        for name, rows in test.items():
            if not located.agrees(name, ids[name], rows.labels):
                raise ValueError(f"the split and the saved probabilities disagree on the words of {name}")
        context, word = finishes.scored(test)
        return cls(Occurrences(context, place), finishes.prior(), finishes.context.tau, word, spoken)

    @staticmethod
    def parser(description: str | None) -> argparse.ArgumentParser:
        """The arguments every study of a blend's scene takes: the blend and the device its data is read on."""
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("name", help="a blend under models/ whose metrics.json lists members with probs.pt")
        parser.add_argument("--device", default=Device.CPU)
        return parser

    @classmethod
    def opened(cls, parser: argparse.ArgumentParser) -> tuple[DataRoot, argparse.Namespace, "Scene"]:
        """The data root, the parsed arguments and the named blend's scene, logging switched on."""
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        return root, args, cls.of(Placed(root, args.name, torch.device(args.device)))


class Report:
    """Logs the study's findings and saves them."""

    def __init__(self, scene: Scene, attributes: Attributes, out: Path) -> None:
        self.scene, self.attributes, self.out = scene, attributes, out

    def run(self) -> dict[str, object]:
        occurrences = self.scene.occurrences
        scored = occurrences.scored
        macro = scored.macro()
        LOG.info(
            "local-test unseen-span words n=%d tau=%.2f bAcc@%d %s", len(scored.labels), self.scene.tau, TOP, macro
        )
        table = occurrences.by_word(self.scene.prior, self.attributes)
        LOG.info("per word\n%s", table.round(3).to_string())
        ranked = table.dropna(subset="hit10").sort_values("hit10")[["n", "hit10", "top1", "duration", "ioi"]].round(3)
        LOG.info("worst %d\n%s", EXTREMES, ranked.head(EXTREMES).to_string())
        LOG.info("best %d\n%s", EXTREMES, ranked.tail(EXTREMES).to_string())
        correlations = occurrences.correlations(table)
        LOG.info("spearman of word hit rate vs %s", correlations)
        content = self._function_split(table)
        LOG.info("function vs content words (mean per-word hit10, words): %s", content)
        errors = Confusions.of(scored)
        words = PRIMARY.words
        pairs = [(words[t], words[p], c) for t, p, c in errors.top(PAIRS)]
        LOG.info(
            "top-1 errors %d of %d words; most frequent (true, predicted, count): %s",
            errors.counts.sum(),
            len(scored.labels),
            pairs,
        )
        kept = {name: errors.kept(attribute) for name, attribute in self.attributes.named().items()}
        LOG.info("errors inside the same attribute (observed, chance from predicted marginal): %s", kept)
        shape = {
            "position": occurrences.by_position(),
            "interval-next": occurrences.by_interval("gap_next"),
            "interval-prev": occurrences.by_interval("gap_prev"),
        }
        for name, frame in shape.items():
            LOG.info("by %s\n%s", name, frame.round(3).to_string())
        record: dict[str, object] = {
            "macro": macro,
            "n": len(scored.labels),
            "tau": self.scene.tau,
            "correlations": correlations,
            "function-vs-content": content,
            "words": json.loads(table.to_json(orient="index")),
            "confusions": pairs,
            "kept": kept,
            "buckets": {
                name: json.loads(frame.rename(index=str).to_json(orient="index")) for name, frame in shape.items()
            },
        }
        self.out.write_text(json.dumps(record, indent=1))
        return record

    def _function_split(self, table: pd.DataFrame) -> dict[str, tuple[float, int]]:
        held = table.dropna(subset="hit10")
        return {
            ("function" if flag else "content"): (float(group["hit10"].mean()), len(group))
            for flag, group in held.groupby("function")
        }

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        root, args, scene = Scene.opened(Scene.parser(__doc__))
        Report(scene, Attributes.of(root), root.models() / args.name / OUT).run()


if __name__ == "__main__":
    Report.main()

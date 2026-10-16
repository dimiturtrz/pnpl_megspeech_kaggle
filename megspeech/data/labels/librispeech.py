"""LibriSpeech word onsets cut into sentence spans: a pretraining corpus for the timing-only model.

    uv run --no-sync python -m megspeech.data.labels.librispeech --fetch   # alignments + metadata, into raw/
    uv run --no-sync python -m megspeech.data.labels.librispeech           # build processed/librispeech_alignments

Sources, each CC BY 4.0 (licence read on the primary page):

- word alignments from the Montreal Forced Aligner, `gilkeyio/librispeech-alignments` on HuggingFace
  (https://huggingface.co/datasets/gilkeyio/librispeech-alignments, dataset card `license: cc-by-4.0`). Only its `id`,
  `transcript` and `words` columns are read, by parquet column range requests; the audio is never downloaded. The
  CorentinJ/librispeech-alignments archive (https://github.com/CorentinJ/librispeech-alignments) is the same MFA run,
  but its Google Drive links are dead and the repo states no licence.
- LibriSpeech-PC manifests with punctuation restored (https://www.openslr.org/145/, CC BY 4.0): where a sentence ends.
- LibriSpeech chapter and author metadata (https://www.openslr.org/12/ `raw-metadata`, CC BY 4.0): which book a
  chapter belongs to, and the books by Arthur Conan Doyle, whose chapters are left out because our test stimuli
  are Sherlock Holmes.

An utterance whose words match the LibriSpeech-PC text is cut at `. ? !`; any other is cut at the alignment's pauses
longer than a threshold calibrated on the matched dev utterances (the F1-best separator of sentence ends from the
other word gaps).
"""

import argparse
import json
import logging
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from jaxtyping import Bool, Float

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import ABSENT, PRIMARY
from megspeech.data.raw.librispeech_alignments import DATASET, Fetch

LOG = logging.getLogger(__name__)

OUTPUT = "spans.parquet"
UNKNOWN = "<unk>"
"""What the aligner writes for a word outside its lexicon."""

CALIBRATION_SPLITS = ("dev_clean", "dev_other")
PAUSE_GRID = tuple(round(0.05 * step, 2) for step in range(1, 21))
"""Candidate pause thresholds in seconds, 50 ms to 1 s."""

EXCLUDED_AUTHOR = ("Conan", "Doyle")
"""First-name fragment and last name of the author whose books are left out."""

MIN_WORDS = 2
CLOSERS = "\"'”’)]}"
ENDINGS = (".", "?", "!")
DASHES = re.compile("[-‐-―]+")
KEPT = re.compile(r"[^a-z0-9']")
ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "jr",
        "sr",
        "capt",
        "col",
        "gen",
        "lieut",
        "messrs",
        "mme",
        "rev",
        "hon",
        "prof",
        "vs",
    }
)
"""Words that end in a full stop without ending a sentence."""
WORD_LETTERS = frozenset({"i", "a"})
"""One-letter words; any other single letter before a full stop is an initial."""


@dataclass(frozen=True)
class Word:
    """One aligned word: its onset and end in seconds, and its lower-case text (`UNKNOWN` if the aligner had none)."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Utterance:
    """One aligned LibriSpeech utterance, its words in order."""

    uid: str
    words: tuple[Word, ...]

    @classmethod
    def from_row(cls, uid: str, transcript: str, aligned: Sequence[Mapping[str, float | str]]) -> "Utterance":
        """The aligner's words, their unknowns restored from the transcript when the two agree on the word count."""
        heard = transcript.lower().split()
        restore = len(heard) == len(aligned)
        words = tuple(
            Word(float(a["start"]), float(a["end"]), heard[i] if restore else str(a["word"]).lower())
            for i, a in enumerate(aligned)
        )
        return cls(uid, words)

    @property
    def speaker(self) -> int:
        return int(self.uid.split("-")[0])

    @property
    def chapter(self) -> int:
        return int(self.uid.split("-")[1])

    def gaps(self) -> list[float]:
        """The silence before each word after the first."""
        return [b.start - a.end for a, b in pairwise(self.words)]


@dataclass(frozen=True)
class Punctuated:
    """The LibriSpeech-PC text of an utterance as words, each flagged when a sentence ends on it."""

    words: tuple[str, ...]
    ends: tuple[bool, ...]

    @classmethod
    def parse(cls, text: str) -> "Punctuated":
        words: list[str] = []
        ends: list[bool] = []
        for raw in text.split():
            pieces = [KEPT.sub("", p.lower().replace("’", "'")).strip("'") for p in DASHES.split(raw)]
            pieces = [p for p in pieces if p]
            if not pieces:
                continue
            words.extend(pieces)
            ends.extend([False] * (len(pieces) - 1) + [cls.closes(raw, pieces[-1])])
        return cls(tuple(words), tuple(ends))

    @staticmethod
    def closes(raw: str, last: str) -> bool:
        """A sentence ends on this token: `. ? !` (closing quotes aside), unless it is an abbreviation or initial."""
        if not raw.rstrip(CLOSERS).endswith(ENDINGS):
            return False
        full_stop = raw.rstrip(CLOSERS).endswith(".")
        initial = len(last) == 1 and last not in WORD_LETTERS
        return not (full_stop and (last in ABBREVIATIONS or initial))

    def matches(self, utterance: Utterance) -> bool:
        """The same words in the same order, an unknown aligned word standing for any."""
        return len(self.words) == len(utterance.words) and all(
            w.text in (UNKNOWN, p) or w.text.strip("'") == p for w, p in zip(utterance.words, self.words, strict=True)
        )


@dataclass(frozen=True)
class Cutter:
    """Cuts an utterance into sentences: by punctuation when its text matches, else at pauses over `pause` seconds."""

    pause: float

    def cut(self, utterance: Utterance, text: Punctuated | None) -> tuple[list[range], bool]:
        """The word ranges of its sentences, and whether punctuation (not a pause) decided them."""
        if text is not None and text.matches(utterance):
            return self.ranges((i + 1 for i, end in enumerate(text.ends) if end), len(text.words)), True
        gaps = utterance.gaps()
        return self.ranges((i + 1 for i, gap in enumerate(gaps) if gap > self.pause), len(utterance.words)), False

    @staticmethod
    def ranges(ends: Iterable[int], size: int) -> list[range]:
        bounds = sorted({0, size, *ends})
        return [range(a, b) for a, b in pairwise(bounds)]


@dataclass(frozen=True)
class Chapters:
    """Which book each chapter belongs to, and which chapters are the excluded author's."""

    book: Mapping[int, int]
    barred: frozenset[int]

    @classmethod
    def read(cls, chapters: Path, annotated: Path, author: tuple[str, str] = EXCLUDED_AUTHOR) -> "Chapters":
        book: dict[int, int] = {}
        project: dict[int, int] = {}
        for line in chapters.read_text(encoding="utf-8").splitlines():
            if line.startswith(";") or not line.strip():
                continue
            fields = [f.strip() for f in line.split("|")]
            book[int(fields[0])] = int(fields[5])
            project[int(fields[0])] = int(fields[4])
        with sqlite3.connect(f"{annotated.as_uri()}?mode=ro", uri=True) as db:
            projects = {
                row[0]
                for row in db.execute(
                    "SELECT x.project_id FROM project_author_associations x JOIN authors a ON a.id = x.author_id "
                    "WHERE a.last_name = ? AND a.first_name LIKE ?",
                    (author[1], f"%{author[0]}%"),
                )
            }
        return cls(book, frozenset(c for c, p in project.items() if p in projects))

    def excluded(self, chapter: int) -> bool:
        return chapter in self.barred


@dataclass
class Tally:
    """What the build kept and dropped, in utterances and words."""

    counts: Counter[str] = field(default_factory=Counter)

    def add(self, kind: str, words: int) -> None:
        self.counts[f"{kind}_utterances"] += 1
        self.counts[f"{kind}_words"] += words

    def share(self, kind: str, of: Iterable[str]) -> float:
        total = sum(self.counts[f"{k}_words"] for k in of)
        return self.counts[f"{kind}_words"] / total if total else 0.0


SCHEMA = pa.schema(
    [
        ("span", pa.int32()),
        ("utterance", pa.string()),
        ("speaker", pa.int32()),
        ("book", pa.int32()),
        ("word", pa.string()),
        ("onset", pa.float32()),
    ]
)
"""One row per word; onset in seconds from the span's first word."""


@dataclass
class Rows:
    """One shard's output columns, filled span by span."""

    columns: dict[str, list[int | float | str]] = field(default_factory=lambda: {name: [] for name in SCHEMA.names})

    def add(self, span: int, utterance: Utterance, sentence: range, book: int) -> None:
        words = [utterance.words[i] for i in sentence]
        origin = words[0].start
        for name, value in (
            ("span", [span] * len(words)),
            ("utterance", [utterance.uid] * len(words)),
            ("speaker", [utterance.speaker] * len(words)),
            ("book", [book] * len(words)),
            ("word", [w.text for w in words]),
            ("onset", [w.start - origin for w in words]),
        ):
            self.columns[name].extend(value)

    def table(self) -> pa.Table:
        return pa.table(self.columns, schema=SCHEMA)


class Corpus:
    """Builds the span table from the fetched shards."""

    def __init__(self, root: DataRoot, *, min_words: int = MIN_WORDS, pause: float | None = None) -> None:
        self.fetch = Fetch(root)
        self.out = root.processed(DATASET)
        self.min_words = min_words
        self.pause = pause

    def utterances(self, shard: Path) -> list[Utterance]:
        table = pq.read_table(shard)
        return [
            Utterance.from_row(uid, transcript, words)
            for uid, transcript, words in zip(
                table.column("id").to_pylist(),
                table.column("transcript").to_pylist(),
                table.column("words").to_pylist(),
                strict=True,
            )
        ]

    def texts(self, shard: Path) -> dict[str, Punctuated]:
        split = shard.name.split("-0")[0].replace("_", "-")
        manifest = self.fetch.meta / f"{split}.json"
        rows = (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
        return {Path(r["audio_filepath"]).stem: Punctuated.parse(r["text"]) for r in rows}

    def threshold(self, shards: Sequence[Path]) -> float:
        """The pause that best separates sentence ends from other gaps in the matched calibration utterances."""
        gaps: list[float] = []
        ends: list[bool] = []
        for shard in shards:
            texts = self.texts(shard)
            for utterance in self.utterances(shard):
                text = texts.get(utterance.uid)
                if text is not None and text.matches(utterance):
                    gaps.extend(utterance.gaps())
                    ends.extend(text.ends[:-1])
        return Calibration(np.array(gaps), np.array(ends, dtype=bool)).best()

    def build(self) -> Path:
        """Write every kept span's words to `spans.parquet`, spans in shard order."""
        shards = self.fetch.shards()
        calibrated = [s for s in shards if s.name.startswith(CALIBRATION_SPLITS)]
        pause = self.pause if self.pause is not None else self.threshold(calibrated)
        LOG.info("pause threshold %.2f s", pause)
        meta = self.fetch.meta
        chapters = Chapters.read(meta / "CHAPTERS.TXT", meta / "lv-annotated.sqlite.db")
        cutter = Cutter(pause)
        tally = Tally()
        target = self.out / OUTPUT
        part = target.with_suffix(".part")
        span = 0
        with pq.ParquetWriter(part, SCHEMA, compression="zstd") as writer:
            for shard in shards:
                rows, span = self.rows(shard, cutter, chapters, tally, span)
                writer.write_table(rows.table())
                LOG.info("%s: %d spans so far", shard.name, span)
        part.replace(target)
        Report(target, tally, chapters).log()
        return target

    def rows(self, shard: Path, cutter: Cutter, chapters: Chapters, tally: Tally, span: int) -> tuple[Rows, int]:
        """`shard`'s kept spans numbered from `span`, and the next free span number; every word lands in `tally`."""
        texts = self.texts(shard)
        rows = Rows()
        for utterance in self.utterances(shard):
            if chapters.excluded(utterance.chapter):
                tally.add("excluded", len(utterance.words))
                continue
            text = texts.get(utterance.uid)
            ranges, punctuated = cutter.cut(utterance, text)
            tally.add("matched" if punctuated else "mismatched" if text else "absent", len(utterance.words))
            for sentence in ranges:
                if len(sentence) < self.min_words:
                    tally.add("short", len(sentence))
                    continue
                rows.add(span, utterance, sentence, chapters.book[utterance.chapter])
                tally.add("kept", len(sentence))
                span += 1
        return rows, span


@dataclass(frozen=True)
class Calibration:
    """Gap lengths and whether a sentence ended in each."""

    gaps: Float[np.ndarray, " n"]
    ends: Bool[np.ndarray, " n"]

    def best(self) -> float:
        """The grid pause whose `gap > pause` rule detects sentence ends with the best F1."""
        scores = [self.f1(self.gaps > pause) for pause in PAUSE_GRID]
        return PAUSE_GRID[int(np.argmax(scores))]

    def f1(self, predicted: Bool[np.ndarray, " n"]) -> float:
        hits = float(np.sum(predicted & self.ends))
        return 2 * hits / max(float(predicted.sum() + self.ends.sum()), 1.0)


@dataclass(frozen=True)
class Report:
    """The built table's statistics, logged."""

    path: Path
    tally: Tally
    chapters: Chapters

    def log(self) -> None:
        counts = self.tally.counts
        decided = ("matched", "mismatched", "absent")
        for kind in decided:
            LOG.info(
                "%s: %d utterances, %.2f%% of words",
                kind,
                counts[f"{kind}_utterances"],
                100 * self.tally.share(kind, decided),
            )
        LOG.info(
            "excluded (%s %s): %d of %d chapters listed, %d utterances, %d words",
            *EXCLUDED_AUTHOR,
            len(self.chapters.barred),
            len(self.chapters.book),
            counts["excluded_utterances"],
            counts["excluded_words"],
        )
        LOG.info("spans under the minimum dropped: %d words", counts["short_words"])
        table = pq.read_table(self.path, columns=["span", "word"])
        sizes = np.bincount(table.column("span").to_numpy())
        LOG.info("spans %d, words %d", len(sizes), table.num_rows)
        LOG.info("words/span median %.0f p10 %.0f p90 %.0f", *np.percentile(sizes, [50, 10, 90]))
        LOG.info("%s share of words: %.2f%%", UNKNOWN, 100 * self.share(table, UNKNOWN))
        found = {w: n for w, n in self.vocabulary(table).items() if n}
        LOG.info(
            "PRIMARY vocabulary: %d of %d words present, %d tokens; counts %s",
            len(found),
            len(PRIMARY),
            sum(found.values()),
            found,
        )

    @staticmethod
    def share(table: pa.Table, word: str) -> float:
        counted = {c["values"]: c["counts"] for c in table.column("word").value_counts().to_pylist()}
        return counted.get(word, 0) / table.num_rows

    @staticmethod
    def vocabulary(table: pa.Table) -> dict[str, int]:
        counted = table.column("word").value_counts().to_pylist()
        spelled = PRIMARY.labels(c["values"] for c in counted)
        totals = Counter[str]()
        for c, label in zip(counted, spelled, strict=True):
            if label != ABSENT:
                totals[PRIMARY.words[label]] += c["counts"]
        return {w: totals[w] for w in PRIMARY.words}


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="download alignments and metadata instead of building")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument(
        "--pause", type=float, default=None, help="fallback pause threshold in seconds (default: calibrated)"
    )
    args = parser.parse_args(argv)
    root = DataRoot.from_config()
    if args.fetch:
        fetch = Fetch(root)
        fetch.metadata()
        fetch.alignments(HfFileSystem())
        return
    Corpus(root, min_words=args.min_words, pause=args.pause).build()


if __name__ == "__main__":
    main()

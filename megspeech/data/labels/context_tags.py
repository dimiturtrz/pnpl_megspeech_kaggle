"""Every heard word's part of speech in the sentence it was said in, one class per token rather than per word type.

Each run's sentence words (ordered by `wordidx` within `sentenceidx`) are handed to `en_core_web_trf` as one
pre-tokenised sentence on the GPU, so a word's tag sees its neighbours: "like" is a verb in one sentence and an
adposition in the next. Pre-tokenising keeps spaCy from splitting a word ("it's" stays one token), so every tag is
aligned to our word and no punctuation token is ever counted as one.

The classes are `CLASSES`: spaCy's universal tags plus `OTHER`, a real class that takes its share of every
distribution. PROPN, NUM, INTJ, X, PUNCT, SYM and SPACE fall into `OTHER`: none contains one of the 50 targets and each
is rare, so each can only say "definitely not a target", which one class says as well as seven.

`UNKNOWN` (-1, the ignore index) is not a class: it marks a token the tagger cannot be trusted on, a "no hint, work it
out yourself". Probes neither train nor score on it, and a fusion treats its mass as neutral. The transformer
pipeline exposes no per-tag probability, so trust is read from the tagger's own consistency across the text instead:

- no tag: spaCy returned a tag count other than the sentence's word count;
- no dominant class: fewer than `DOMINANT` of the word type's tokens share its most frequent class, so its class is a
  property of the sentence, not of the word, and no single class speaks for it;
- a minor reading: the token carries a class under `MINOR` of its type's tokens, a tagger slip beside an otherwise
  settled word.

A type seen on fewer than `SETTLED` tokens has too few for either share to mean anything and keeps its class. A word
outside any sentence has no context and reads as `UNKNOWN`. The cache is keyed by (run, word onset), the key a split's
words carry, and read from subject 0's training and validation runs only: the Kaggle test carries MEG alone.

    uv run python -m megspeech.data.labels.context_tags   # under processed/syntax
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import spacy
from jaxtyping import Float, Int, Shaped
from spacy.language import Language
from spacy.tokens import Doc

from core.paths import DataRoot
from megspeech.data.constants import MISSING
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import PRIMARY, Vocabulary
from megspeech.data.raw.runs import DEEP, Partition, Run

LOG = logging.getLogger(__name__)

DOMINANT = 0.6
"""The share of a word type's tokens its most frequent class must hold: a clear majority, so the class is the word's."""
MINOR = 0.05
"""The share of a settled type's tokens a tag falls under to count as a tagger slip, not a reading."""
SETTLED = 20
"""Tokens a type needs before its tag shares are judged: one slip in fewer than 20 is already over `MINOR`."""
BATCH = 128
"""Sentences spaCy tags per GPU batch."""
SENTENCE_COLUMNS = ["kind", "segment", "sentenceidx", "wordidx", "timemeg"]
CLASSES = [*("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "PART", "PRON", "SCONJ", "VERB"), "OTHER"]
"""The tag set a token is classified into; the last class takes every tag the others do not name."""
OTHER = len(CLASSES) - 1
TAGGED_COLUMNS = ["run", "onset", "word", "tag"]
SUMMARY_TAGS = 3
"""Classes each vocabulary word's summary lists, by share."""


class Tagger(Protocol):
    def tag(self, sentences: Sequence[list[str]]) -> list[list[str]]:
        """Each pre-tokenised sentence's universal tags, one per word."""
        ...


@dataclass(frozen=True)
class SpacyTagger:
    """`en_core_web_trf` tagging pre-tokenised sentences, on the GPU when `gpu`."""

    nlp: Language

    @classmethod
    def load(cls, *, gpu: bool = True) -> "SpacyTagger":  # devtools-ignore: test-mirror
        if gpu:
            spacy.require_gpu()
        return cls(spacy.load("en_core_web_trf", exclude=["parser", "ner", "lemmatizer"]))

    def tag(self, sentences: Sequence[list[str]]) -> list[list[str]]:
        docs = self.nlp.pipe((Doc(self.nlp.vocab, words=words) for words in sentences), batch_size=BATCH)
        return [[token.pos_ for token in doc] for doc in docs]


@dataclass(frozen=True)
class ContextTags:
    """The class (an index into `CLASSES`) of every sentence word tagged, keyed by its run and onset."""

    runs: Shaped[np.ndarray, " k"]
    onsets: Float[np.ndarray, " k"]
    tags: Int[np.ndarray, " k"]

    @staticmethod
    def words(run: Run) -> pd.DataFrame:
        """One run's sentence words in sentence order: `run`, `onset`, `sentence` and the `word` as spoken."""
        table = pd.read_csv(run.events, sep="\t", usecols=SENTENCE_COLUMNS, keep_default_na=False, na_values=MISSING)
        heard = table.loc[(table["kind"] == "word") & table["segment"].notna()].dropna(
            subset=["sentenceidx", "wordidx"]
        )
        ordered = heard.sort_values(["sentenceidx", "wordidx"])
        return pd.DataFrame(
            {
                "run": run.name,
                "onset": ordered["timemeg"].to_numpy(np.float64),
                "sentence": ordered["sentenceidx"].to_numpy(np.int64),
                "word": ordered["segment"].astype(str).to_numpy(),
            }
        )

    @staticmethod
    def tagged(words: pd.DataFrame, tagger: Tagger) -> pd.DataFrame:
        """`words` (`ContextTags.words` of any runs) beside the `tag` its sentence gives it, "" where the tagger's
        output does not match the sentence."""
        groups = [group for _, group in words.groupby(["run", "sentence"], sort=False)]
        given = tagger.tag([group["word"].tolist() for group in groups])
        flat = chain.from_iterable(
            tags if len(tags) == len(group) else [""] * len(group) for group, tags in zip(groups, given, strict=True)
        )
        return words.loc[pd.concat(groups).index].assign(tag=list(flat))[TAGGED_COLUMNS].reset_index(drop=True)

    @staticmethod
    def classified(tagged: pd.DataFrame) -> Int[np.ndarray, " n"]:
        """Each token's class: its tag in `CLASSES` (`OTHER` for any other), `UNKNOWN` where it has no tag, its type's
        tokens have no dominant class or it is a minor reading of a settled type."""
        index = pd.Series(range(len(CLASSES)), index=CLASSES)
        coded = tagged["tag"].map(index).fillna(OTHER).where(tagged["tag"] != "", UNKNOWN).astype(np.int64)
        frame = pd.DataFrame({"type": Vocabulary.token(tagged["word"]), "code": coded.to_numpy()}, index=tagged.index)
        total = frame.groupby("type")["code"].transform("size")
        share = frame.groupby(["type", "code"])["code"].transform("size") / total
        modal = share.groupby(frame["type"]).transform("max")
        trusted = (total < SETTLED) | ((modal >= DOMINANT) & (share >= MINOR))
        return coded.where(trusted, UNKNOWN).to_numpy()

    @classmethod
    def build(cls, tagged: pd.DataFrame) -> "ContextTags":
        """The table of `tagged`'s classes, sorted by run and onset."""
        table = tagged.assign(code=cls.classified(tagged)).sort_values(["run", "onset"])
        table = table.drop_duplicates(["run", "onset"])
        return cls(table["run"].to_numpy(str), table["onset"].to_numpy(np.float64), table["code"].to_numpy(np.int64))

    @classmethod
    def read(cls, runs: Sequence[Run], tagger: Tagger) -> "ContextTags":
        return cls.build(cls.tagged(pd.concat([cls.words(run) for run in runs], ignore_index=True), tagger))

    @staticmethod
    def summary(tagged: pd.DataFrame, vocabulary: Vocabulary, *, top: int = SUMMARY_TAGS) -> pd.DataFrame:
        """Each vocabulary word's token count and its `top` classes by share (`UNKNOWN` included), over `tagged`."""
        named = pd.Series(np.array([*CLASSES, "UNKNOWN"])[ContextTags.classified(tagged)], index=tagged.index)
        spelled = Vocabulary.spelled(tagged["word"])
        rows = []
        for word in vocabulary.words:
            mine = spelled == word
            shares = named[mine].value_counts(normalize=True).head(top)
            rows.append(
                {"word": word, "tokens": int(mine.sum()), "classes": {k: round(float(v), 3) for k, v in shares.items()}}
            )
        return pd.DataFrame(rows)

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "syntax" / "context_tags.npz"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, runs=self.runs, onsets=self.onsets, tags=self.tags)
        return path

    @classmethod
    def load(cls, path: Path) -> "ContextTags":
        with np.load(path) as saved:
            return cls(saved["runs"], saved["onsets"], saved["tags"])

    def classes(self, runs: Shaped[np.ndarray, " n"], onsets: Float[np.ndarray, " n"]) -> Int[np.ndarray, " n"]:
        """Each (run, onset) word's class, `UNKNOWN` for a word the table lacks."""
        keys = pd.MultiIndex.from_arrays([self.runs, self.onsets])
        at = keys.get_indexer(pd.MultiIndex.from_arrays([runs, onsets]))
        return np.where(at >= 0, self.tags[at.clip(min=0)], UNKNOWN)

    @staticmethod
    def trained(runs: Sequence[Run]) -> list[Run]:
        """Subject 0's runs whose text may be tagged: every run but the held-out test."""
        return [run for run in runs if run.subject == DEEP and run.partition is not Partition.TEST]

    @classmethod
    def main(cls) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        runs = cls.trained(Run.discover(root.raw()))
        tagged = cls.tagged(pd.concat([cls.words(run) for run in runs], ignore_index=True), SpacyTagger.load())
        table = cls.build(tagged)
        counts = pd.Series(np.array([*CLASSES, "UNKNOWN"])[table.tags]).value_counts().to_dict()
        LOG.info("%d tokens of %d runs -> %s", len(table.tags), len(runs), table.save(cls.path(root)))
        LOG.info("class counts %s, UNKNOWN %.4f", counts, (table.tags == UNKNOWN).mean())
        LOG.info("untagged by spaCy %.4f", (tagged["tag"] == "").mean())
        LOG.info("vocabulary words in context:\n%s", cls.summary(tagged, PRIMARY).to_string())


if __name__ == "__main__":
    ContextTags.main()

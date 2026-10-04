"""Every heard word's part of speech, tagged by spaCy within the sentence the events tables place it in.

Each run's word rows, ordered by `wordidx` within each `sentenceidx`, are tagged as one pre-tokenised sentence with
`en_core_web_sm`'s universal (UPOS) tagger, so a word's tag sees its neighbours. A word type takes its most frequent
tag over every sentence it was heard in: "like" is a verb or an adposition by context, and the table keeps one. The
Kaggle test carries MEG alone, so the table can only be a training target: a class for every word the 50 leave out.

    uv run python -m megspeech.data.labels.syntax   # every downloaded events file's words, under processed/syntax
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import en_core_web_sm
import numpy as np
import pandas as pd
import torch
from jaxtyping import Int
from spacy.language import Language
from spacy.tokens import Doc
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.constants import MISSING
from megspeech.data.labels.phonemes import EVENTS, UNKNOWN, WORD
from megspeech.data.labels.vocabulary import Vocabulary

LOG = logging.getLogger(__name__)

TAGS = [
    *("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM"),
    *("PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"),
]
"""The 17 universal parts of speech (Universal Dependencies v2) spaCy's `pos_` takes."""


@dataclass(frozen=True)
class PartsOfSpeech:
    """The part of speech (an index into `TAGS`) of every distinct word heard, keyed by the `token` a split carries
    (`Vocabulary.token`), sorted."""

    keys: Int[Tensor, " k"]
    tags: Int[Tensor, " k"]

    @staticmethod
    def tagger() -> Language:
        return en_core_web_sm.load(disable=["parser", "ner", "lemmatizer"])

    @staticmethod
    def sentences(events: Path) -> list[list[str]]:
        """One run's sentences, each its words in order; words outside a sentence left out."""
        table = pd.read_csv(
            events, sep="\t", usecols=["kind", "segment", *WORD], keep_default_na=False, na_values=MISSING
        )
        words = table.loc[(table["kind"] == "word") & table["segment"].notna()].dropna(subset=WORD)
        ordered = words.sort_values(WORD)
        return [group["segment"].astype(str).tolist() for _, group in ordered.groupby("sentenceidx", sort=True)]

    @staticmethod
    def tagged(sentences: Sequence[list[str]], tagger: Language) -> pd.DataFrame:
        """Every word of `sentences` beside its tag in that sentence."""
        docs = tagger.pipe(Doc(tagger.vocab, words=words) for words in sentences)
        return pd.DataFrame([(token.text, token.pos_) for doc in docs for token in doc], columns=["word", "tag"])

    @classmethod
    def build(cls, tagged: pd.DataFrame) -> "PartsOfSpeech":
        """The table of (word, tag) rows: each token at its most frequent tag among `TAGS`, ties to the earlier."""
        index = pd.Series(range(len(TAGS)), index=TAGS)
        coded = tagged.assign(code=tagged["tag"].map(index)).dropna(subset="code")
        known = coded.assign(token=Vocabulary.token(coded["word"]), code=coded["code"].astype(np.int64))
        counted = known.groupby(["token", "code"]).size().reset_index(name="count")
        chosen = counted.sort_values(["token", "count", "code"], ascending=[True, False, True]).drop_duplicates("token")
        return cls(
            torch.from_numpy(chosen["token"].to_numpy(np.int64, copy=True)),
            torch.from_numpy(chosen["code"].to_numpy(np.int64, copy=True)),
        )

    @classmethod
    def read(cls, events: Sequence[Path], tagger: Language) -> "PartsOfSpeech":
        sentences = [sentence for path in events for sentence in cls.sentences(path)]
        return cls.build(cls.tagged(sentences, tagger))

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "syntax" / "tags.npz"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys.numpy(), tags=self.tags.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "PartsOfSpeech":
        with np.load(path) as saved:
            return cls(torch.from_numpy(saved["keys"]), torch.from_numpy(saved["tags"]))

    def to(self, device: torch.device) -> "PartsOfSpeech":
        return PartsOfSpeech(self.keys.to(device), self.tags.to(device))

    def classes(self, token: Int[Tensor, "*b"]) -> Int[Tensor, "*b"]:
        """Each token's tag, `UNKNOWN` for a token the table lacks."""
        at = torch.searchsorted(self.keys, token).clamp(max=len(self.keys) - 1)
        return torch.where(self.keys[at] == token, self.tags[at], UNKNOWN)

    @classmethod
    def main(cls) -> None:
        """Write the table of every subject-0 events file downloaded, its MEG downloaded or not."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        events = [path for path in sorted(root.raw().rglob(EVENTS)) if path.stat().st_size]
        table = cls.read(events, cls.tagger())
        LOG.info("%d distinct words from %d runs -> %s", len(table.keys), len(events), table.save(cls.path(root)))


if __name__ == "__main__":
    PartsOfSpeech.main()

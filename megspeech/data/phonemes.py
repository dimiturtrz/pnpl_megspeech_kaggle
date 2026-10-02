"""Every heard word's first phoneme, read from the forced alignment the events tables already carry.

Each run's events list the phonemes of every word (`kind == "phoneme"`, labelled `ah_B`: ARPAbet without stress,
then the position in the word), joined to their word by `sentenceidx` and `wordidx` and ordered by `phonemeidx`.
The alignment comes from a pronunciation lexicon, so a word's first phoneme is a property of its text: of 23,129
distinct words, 165 begin on more than one, and each is kept at its most frequent. The holdout carries MEG alone,
so the table can only be a training target.

    uv run python -m megspeech.data.phonemes      # every downloaded events file's words, under processed/phonemes
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.continuous import MISSING
from megspeech.data.corpus import Corpus

LOG = logging.getLogger(__name__)

PHONEMES = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
    "zh",
]
"""The 39 ARPAbet phonemes the alignment uses; its `oov` and `sil` name no sound and are left out."""

UNKNOWN = -1
"""The phoneme of a token the table lacks, or whose first phoneme is `oov`."""

WORD = ["sentenceidx", "wordidx"]
EVENTS = "sub-0_*_events.tsv"


@dataclass(frozen=True)
class Pronunciation:
    """The first phoneme (an index into `PHONEMES`) of every distinct word heard, keyed by the `token` a split
    carries (`Corpus.token`), sorted."""

    keys: Int[Tensor, " k"]
    first: Int[Tensor, " k"]

    @staticmethod
    def initials(events: Path) -> pd.DataFrame:
        """One run's words beside the first phoneme the alignment gives each, `oov` included."""
        table = pd.read_csv(
            events,
            sep="\t",
            usecols=["kind", "segment", *WORD, "phonemeidx"],
            keep_default_na=False,
            na_values=MISSING,
        )
        words = table.loc[(table["kind"] == "word") & table["segment"].notna(), ["segment", *WORD]]
        heads = table.loc[(table["kind"] == "phoneme") & (table["phonemeidx"] == 0), ["segment", *WORD]]
        heads = heads.assign(phoneme=heads["segment"].str.split("_").str[0]).drop(columns="segment")
        joined = words.dropna(subset=WORD).merge(heads.dropna(subset=WORD).drop_duplicates(WORD), on=WORD)
        return joined.rename(columns={"segment": "word"})[["word", "phoneme"]]

    @classmethod
    def build(cls, initials: pd.DataFrame) -> "Pronunciation":
        """The table of (word, phoneme) rows: each token at its most frequent known first phoneme."""
        index = pd.Series(range(len(PHONEMES)), index=PHONEMES)
        known = initials.assign(token=Corpus.token(initials["word"]), first=initials["phoneme"].map(index)).dropna(
            subset="first"
        )
        counted = known.groupby(["token", "first"]).size().reset_index(name="count")
        chosen = counted.sort_values(["token", "count", "first"], ascending=[True, False, True]).drop_duplicates(
            "token"
        )
        return cls(
            torch.from_numpy(chosen["token"].to_numpy(np.int64, copy=True)),
            torch.from_numpy(chosen["first"].to_numpy(np.int64, copy=True)),
        )

    @classmethod
    def read(cls, events: Sequence[Path]) -> "Pronunciation":
        return cls.build(pd.concat([cls.initials(path) for path in events], ignore_index=True))

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "phonemes" / "first.npz"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys.numpy(), first=self.first.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "Pronunciation":
        with np.load(path) as saved:
            return cls(torch.from_numpy(saved["keys"]), torch.from_numpy(saved["first"]))

    def to(self, device: torch.device) -> "Pronunciation":
        return Pronunciation(self.keys.to(device), self.first.to(device))

    def phonemes(self, token: Int[Tensor, "*b"]) -> Int[Tensor, "*b"]:
        """Each token's first phoneme, `UNKNOWN` for a token the table lacks."""
        at = torch.searchsorted(self.keys, token).clamp(max=len(self.keys) - 1)
        return torch.where(self.keys[at] == token, self.first[at], UNKNOWN)

    @classmethod
    def main(cls) -> None:
        """Write the table of every subject-0 events file downloaded, its MEG downloaded or not."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        events = [path for path in sorted(root.raw().rglob(EVENTS)) if path.stat().st_size]
        table = cls.read(events)
        LOG.info("%d distinct words from %d runs -> %s", len(table.keys), len(events), table.save(cls.path(root)))


if __name__ == "__main__":
    Pronunciation.main()

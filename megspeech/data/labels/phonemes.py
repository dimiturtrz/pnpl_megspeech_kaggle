"""Every heard word's first phonemes, read from the forced alignment the events tables already carry.

Each run's events list the phonemes of every word (`kind == "phoneme"`, labelled `ah_B`: ARPAbet without stress,
then the position in the word), joined to their word by `sentenceidx` and `wordidx` and ordered by `phonemeidx`.
The alignment comes from a pronunciation lexicon, so a word's spelling is a property of its text: of 23,129
distinct words, 165 begin on more than one phoneme, and each is kept at its most frequent spelling. The holdout
carries MEG alone, so the table can only be a training target.

    uv run python -m megspeech.data.labels.phonemes   # every downloaded events file's words, under processed/phonemes
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int
from pnpl.datasets.constants import ARPABET
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.constants import MISSING
from megspeech.data.labels.vocabulary import Vocabulary

LOG = logging.getLogger(__name__)

PHONEMES = sorted(phoneme.lower() for phoneme in ARPABET)
"""The 39 ARPAbet phonemes the alignment uses, as it spells them; its `oov` and `sil` name no sound and are left out."""

UNKNOWN = -1
"""The phoneme of a token the table lacks, at an `oov` position, or past the end of a shorter word."""

POSITIONS = 4
"""Leading phonemes the table spells each word with: a word has 3.25 on average (Sherlock1 run 1, 1869 words),
and a 0.5 s window hears ~4 phonemes of ~0.1 s past the onset."""

WORD = ["sentenceidx", "wordidx"]
EVENTS = "sub-0_*_events.tsv"


@dataclass(frozen=True)
class Pronunciation:
    """The first `POSITIONS` phonemes (indices into `PHONEMES`, `UNKNOWN` past the end) of every distinct word
    heard, keyed by the `token` a split carries (`Vocabulary.token`), sorted."""

    keys: Int[Tensor, " k"]
    spelled: Int[Tensor, "k p"]

    @staticmethod
    def spellings(events: Path) -> pd.DataFrame:
        """One run's words beside the phonemes the alignment gives each at positions 0..`POSITIONS`-1, `oov`
        included and missing positions NaN."""
        table = pd.read_csv(
            events,
            sep="	",
            usecols=["kind", "segment", *WORD, "phonemeidx"],
            keep_default_na=False,
            na_values=MISSING,
        )
        words = table.loc[(table["kind"] == "word") & table["segment"].notna(), ["segment", *WORD]]
        sounds = table.loc[(table["kind"] == "phoneme") & (table["phonemeidx"] < POSITIONS)].dropna(subset=WORD)
        sounds = sounds.assign(phoneme=sounds["segment"].str.split("_").str[0], at=sounds["phonemeidx"].astype(int))
        spelled = sounds.drop_duplicates([*WORD, "at"]).pivot(index=WORD, columns="at", values="phoneme")
        spelled = spelled.reindex(columns=range(POSITIONS)).reset_index()
        joined = words.dropna(subset=WORD).merge(spelled, on=WORD)
        return joined.rename(columns={"segment": "word"})[["word", *range(POSITIONS)]]

    @classmethod
    def build(cls, spellings: pd.DataFrame) -> "Pronunciation":
        """The table of (word, phoneme per position) rows: each token at its most frequent spelling with a known
        first phoneme."""
        index = pd.Series(range(len(PHONEMES)), index=PHONEMES)
        positions = list(range(POSITIONS))
        mapped = spellings.reindex(columns=["word", *positions])
        coded = mapped.assign(**{str(at): mapped[at].map(index) for at in positions}).dropna(subset="0")
        columns = [str(at) for at in positions]
        known = (
            coded.assign(token=Vocabulary.token(coded["word"]))[["token", *columns]].fillna(UNKNOWN).astype(np.int64)
        )
        counted = known.groupby(["token", *columns]).size().reset_index(name="count")
        chosen = counted.sort_values(
            ["token", "count", *columns], ascending=[True, False, *[True] * POSITIONS]
        ).drop_duplicates("token")
        return cls(
            torch.from_numpy(chosen["token"].to_numpy(np.int64, copy=True)),
            torch.from_numpy(chosen[columns].to_numpy(np.int64, copy=True)),
        )

    @classmethod
    def read(cls, events: Sequence[Path]) -> "Pronunciation":
        return cls.build(pd.concat([cls.spellings(path) for path in events], ignore_index=True))

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "phonemes" / "spelled.npz"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys.numpy(), spelled=self.spelled.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "Pronunciation":
        with np.load(path) as saved:
            return cls(torch.from_numpy(saved["keys"]), torch.from_numpy(saved["spelled"]))

    def to(self, device: torch.device) -> "Pronunciation":
        return Pronunciation(self.keys.to(device), self.spelled.to(device))

    def phonemes(self, token: Int[Tensor, "*b"]) -> Int[Tensor, "*b p"]:
        """Each token's spelling, all `UNKNOWN` for a token the table lacks."""
        at = torch.searchsorted(self.keys, token).clamp(max=len(self.keys) - 1)
        return torch.where((self.keys[at] == token).unsqueeze(-1), self.spelled[at], UNKNOWN)

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

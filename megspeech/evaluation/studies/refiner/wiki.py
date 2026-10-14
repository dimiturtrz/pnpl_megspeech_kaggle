"""External English text as the label sequences a span refiner pretrains on: WikiText-103's training articles, each
word mapped to one of the 50 or `OOV`, cut into spans by the training rule (`Spans.by_sentence`).

No competition text may enter: an article naming Sherlock, Conan Doyle or The Moth is dropped whole. WikiText splits
clitics off their word (`it 's`, `do n't`); they are joined back so `it’s` is the vocabulary word it is when heard. A
token with no letter left once spelled (numbers, punctuation) is no heard word and is dropped; `.`, `!` and `?` end a
sentence. The spans are saved under the models root as one `(g, MAX_WORDS)` state table, `PAD` past a span's end.

    uv run python -m megspeech.evaluation.studies refiner.wiki
"""

import argparse
import logging
import re
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import NON_WORD, PRIMARY, VOCAB_SIZE
from megspeech.data.spans.spans import PAD, Spans

LOG = logging.getLogger(__name__)

REPO = "Salesforce/wikitext"
DATASET = "wikitext-103"
FILES = tuple(f"wikitext-103-raw-v1/train-0000{part}-of-00002.parquet" for part in range(2))
EXCLUDED = ("Sherlock", "Conan Doyle", "The Moth")
"""Names of the texts the competition reads: an article carrying one is dropped."""
ENDS = frozenset({".", "!", "?"})
CLITIC = re.compile(r"^('[A-Za-z]+|n't)$")
"""A token WikiText split off the word before it."""
HEADING = re.compile(r"^ = [^=].* = $")
OOV = VOCAB_SIZE
STUDY = "lmrefine"
OUT = "wikispans.pt"


@dataclass(frozen=True)
class Wikitext:
    """WikiText's lines, articles one after another, each opened by its top-level heading."""

    lines: Sequence[str]

    def articles(self) -> list[list[str]]:
        """Every article's lines, its heading first."""
        out: list[list[str]] = []
        for line in self.lines:
            if HEADING.match(line.rstrip("\n")) or not out:
                out.append([])
            out[-1].append(line)
        return out

    def kept(self) -> tuple[list[list[str]], int]:
        """The articles naming none of `EXCLUDED`, and how many were dropped."""
        articles = self.articles()
        kept = [article for article in articles if not any(name in "".join(article) for name in EXCLUDED)]
        return kept, len(articles) - len(kept)


@dataclass(frozen=True)
class Tokens:
    """The vocabulary's spelling of heard words: each one a label of the 50 or `OOV`."""

    positions: dict[str, int]

    @staticmethod
    def words(line: str) -> list[str]:
        """`line`'s tokens with each clitic joined back to its word; a heading line has none."""
        if line.strip().startswith("="):
            return []
        out: list[str] = []
        for token in line.split():
            if out and CLITIC.match(token):
                out[-1] += token
            else:
                out.append(token)
        return out

    def sentences(self, lines: Sequence[str]) -> Iterator[list[int]]:
        """Every sentence's heard words as states, sentences closed by `ENDS` or a line's end."""
        for line in lines:
            sentence: list[int] = []
            for token in self.words(line):
                if token in ENDS:
                    if sentence:
                        yield sentence
                    sentence = []
                    continue
                spelled = NON_WORD.sub("", token.lower().replace("'", "’"))
                if spelled:
                    sentence.append(self.positions.get(spelled, OOV))
            if sentence:
                yield sentence

    def flat(self, articles: Sequence[Sequence[str]]) -> tuple[Int[np.ndarray, " n"], Int[np.ndarray, " s"]]:
        """Every sentence's states end to end, and each sentence's length."""
        states, sizes = array("b"), array("q")
        for article in articles:
            for sentence in self.sentences(article):
                states.extend(sentence)
                sizes.append(len(sentence))
        return np.frombuffer(states, dtype=np.int8), np.frombuffer(sizes, dtype=np.int64)

    @staticmethod
    def spans(states: Int[np.ndarray, " n"], sizes: Int[np.ndarray, " s"]) -> Int[Tensor, "g w"]:
        """Sentences of `sizes` over `states`, cut as training spans are, left-aligned, `PAD` after."""
        group = np.repeat(np.arange(len(sizes)), sizes)
        index = Spans.by_sentence(group, np.zeros(len(group))).index
        return torch.from_numpy(np.where(index != PAD, states[index.clip(min=0)], PAD).astype(np.int8))

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        parser = argparse.ArgumentParser(description=__doc__)
        parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        local = root.raw(DATASET)
        paths = [Path(hf_hub_download(REPO, name, repo_type="dataset", token=False, local_dir=local)) for name in FILES]
        lines = pd.concat([pd.read_parquet(path)["text"] for path in paths]).tolist()
        articles, dropped = Wikitext(lines).kept()
        LOG.info("%d articles kept, %d dropped for naming %s", len(articles), dropped, EXCLUDED)
        states, sizes = Tokens(PRIMARY.positions).flat(articles)
        spans = Tokens.spans(states, sizes)
        LOG.info(
            "%d sentences, %d spans, %d words, in-vocabulary %.4f",
            len(sizes),
            len(spans),
            len(states),
            float((states < OOV).mean()),
        )
        out = root.models() / STUDY
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"spans": spans, "dropped": dropped}, out / OUT)
        LOG.info("wrote %s", out / OUT)


if __name__ == "__main__":
    Tokens.main()

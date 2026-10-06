"""Frozen t5-large embeddings of the 50 words, the semantic targets d'Ascoli et al. align MEG to, and of every
word heard, in or out of the 50 (`--lexicon`).

uv run python -m megspeech.objective.targets.t5 [--lexicon]
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel

from core.paths import DataRoot
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.spans.grouping import Grouping

MODEL = "google-t5/t5-large"
STEM = MODEL.split("/")[-1]
LAYER = 12
"""Hidden state 12 of t5-large's 25: the middle of the encoder, as in d'Ascoli et al. (`layers=0.5`)."""
LOG = logging.getLogger(__name__)


class Encoded(Protocol):
    @property
    def hidden_states(self) -> tuple[Tensor, ...]: ...


Tokenizer = Callable[..., dict[str, Int[Tensor, "1 k"]]]
Encoder = Callable[..., Encoded]


@dataclass(frozen=True)
class Prototypes:
    vectors: Float[Tensor, "v d"]

    @classmethod
    def encode(cls, words: tuple[str, ...], tokenizer: Tokenizer, encoder: Encoder, layer: int) -> "Prototypes":
        """Each word alone, without special tokens, mean-pooled over its tokens at `layer`."""
        rows = []
        with torch.no_grad():
            for word in words:
                ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
                hidden = encoder(input_ids=ids, output_hidden_states=True).hidden_states[layer]
                rows.append(hidden[0].mean(dim=0))
        return cls(torch.stack(rows).float())

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "prototypes" / f"{STEM}-layer{LAYER}.npy"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "Prototypes":
        return cls(torch.from_numpy(np.load(path)))

    @classmethod
    def write(cls, root: DataRoot, tokenizer: Tokenizer, encoder: Encoder) -> Path:
        return cls.encode(PRIMARY.words, tokenizer, encoder, LAYER).save(cls.path(root))

    @staticmethod
    def main() -> None:
        """Download t5-large once and write the 50 words' prototypes, or the lexicon, under processed/prototypes."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--lexicon", action="store_true", help="every heard word, not the 50")
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=False)
        if tokenizer is None:
            raise FileNotFoundError(MODEL)
        encoder = T5EncoderModel.from_pretrained(MODEL, token=False).eval()
        write = Lexicon.write if args.lexicon else Prototypes.write
        LOG.info("wrote %s", write(DataRoot.from_config(), tokenizer, encoder))


@dataclass(frozen=True)
class Lexicon:
    """The t5 embedding of every distinct word heard, in or out of the 50, keyed by the `token` a split carries
    (`Corpus.words`): the targets that let every window, not only the 50 words', train the encoder."""

    keys: Int[Tensor, " k"]
    vectors: Float[Tensor, "k d"]

    @classmethod
    def encode(cls, words: pd.DataFrame, tokenizer: Tokenizer, encoder: Encoder, layer: int) -> "Lexicon":
        """The distinct `token`s of `words` (`Corpus.words` rows), each lower-cased word as `Prototypes.encode`."""
        distinct = words.drop_duplicates("token").sort_values("token")
        text = tuple(distinct["word"].astype(str).str.lower())
        vectors = Prototypes.encode(text, tokenizer, encoder, layer).vectors
        return cls(torch.from_numpy(distinct["token"].to_numpy(np.int64, copy=True)), vectors)

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "prototypes" / f"{STEM}-layer{LAYER}-lexicon.npz"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys.numpy(), vectors=self.vectors.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        with np.load(path) as saved:
            return cls(torch.from_numpy(saved["keys"]), torch.from_numpy(saved["vectors"]))

    def to(self, device: torch.device) -> "Lexicon":
        return Lexicon(self.keys.to(device), self.vectors.to(device))

    def rows(self, token: Int[Tensor, "*b"]) -> Int[Tensor, "*b"]:
        """Each token's row of `vectors`, -1 for a token the lexicon lacks."""
        at = torch.searchsorted(self.keys, token).clamp(max=len(self.keys) - 1)
        return torch.where(self.keys[at] == token, at, -1)

    @classmethod
    def write(cls, root: DataRoot, tokenizer: Tokenizer, encoder: Encoder) -> Path:
        """Every word of every cached run's table, as the sentence path reads them."""
        cache = RunCache.under(root.processed())
        names = sorted(path.stem for path in cache.cache.glob("*.npy"))
        lexicon = cls.encode(Corpus.words(cache.table(names), Grouping.SENTENCE), tokenizer, encoder, LAYER)
        LOG.info("%d distinct words from %d runs", len(lexicon.keys), len(names))
        return lexicon.save(cls.path(root))


if __name__ == "__main__":
    Prototypes.main()
